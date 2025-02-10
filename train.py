import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput

from utils import has_chinese


os.environ["TOKENIZERS_PARALLELISM"] = "false"


LABEL2ID = {"O": 0, "，": 1, "。": 2, "？": 3}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


@dataclass
class ExtraTrainingArguments:
    model_path: str
    dataset_name: str
    project_name: Optional[str] = None


def compute_loss_func(outputs: TokenClassifierOutput, labels: torch.Tensor, num_items_in_batch: int):
    alpha = 1
    gamma = 2
    ignore_index = -100
    num_labels = len(LABEL2ID)

    logits = outputs.logits.view(-1, num_labels)
    labels = labels.view(-1)

    logits = logits[labels != ignore_index]
    labels = labels[labels != ignore_index]
    if len(labels) == 0:
        return torch.Tensor(0)

    log_p = F.log_softmax(logits, dim=-1)
    ce_loss = F.nll_loss(log_p, labels)

    log_pt = log_p[torch.arange(len(logits)), labels]
    pt = log_pt.exp()

    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def preprocess_logits_for_metrics(logits, labels) -> torch.Tensor:
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    preds = eval_preds.predictions.flatten()
    labels = eval_preds.label_ids.flatten()

    metrics_dict = {}
    per_label_precisions, per_label_recalls, per_label_f1s, _ = precision_recall_fscore_support(
        labels, preds, average=None
    )
    label_names = ["O", "comma", "period", "question"]
    for label_name, precision, recall, f1 in zip(label_names, per_label_precisions, per_label_recalls, per_label_f1s):
        if label_name == "O":
            continue
        metrics_dict[f"{label_name}_precision"] = precision
        metrics_dict[f"{label_name}_recall"] = recall
        metrics_dict[f"{label_name}_f1"] = f1
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    metrics_dict["macro_precision"] = macro_precision
    metrics_dict["macro_recall"] = macro_recall
    metrics_dict["macro_f1"] = macro_f1
    _, _, micro_f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    metrics_dict["micro_f1"] = micro_f1
    for k, v in metrics_dict.items():
        metrics_dict[k] = round(v, 4) * 100
    return metrics_dict


def tokenize(examples: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    batch_words_without_punct = []
    batch_labels = []
    for text in examples["text"]:
        words = re.findall(
            r"[a-zA-Z0-9]+|[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎𠻺𠸏𠺢𡃉𡁵，。？]", text
        )
        words_without_punct = []
        labels = []
        for word in words:
            if word in LABEL2ID:
                if labels:
                    labels[-1] = LABEL2ID[word]
            else:
                words_without_punct.append(word)
                labels.append(LABEL2ID["O"])
        batch_words_without_punct.append(words_without_punct)
        batch_labels.append(labels)

    encoded_inputs = tokenizer(batch_words_without_punct, is_split_into_words=True)

    batch_size = len(encoded_inputs["input_ids"])
    batch_aligned_labels = []
    for batch_idx in range(batch_size):
        seq_len = len(encoded_inputs["input_ids"])
        aligned_labels = [-100] * seq_len
        for token_idx in range(seq_len):
            word_idx = encoded_inputs.token_to_word(batch_idx, token_idx)
            if word_idx is not None:
                aligned_labels[token_idx] = batch_labels[batch_idx][word_idx]
        batch_aligned_labels.append(aligned_labels)
    encoded_inputs["labels"] = batch_aligned_labels
    return encoded_inputs


def add_unk_chars(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    unk_chars = []
    token_list_filepath = (
        "/mnt/nas2/asr_models/fano-mdl-online-asr-v3.8.3.2-conformer-baseline-cantonese/data/token_list/char/tokens.txt"
    )
    with open(token_list_filepath, "r") as f:
        for line in f.readlines():
            token = line.strip()
            if len(token) != 1 or not has_chinese(token):
                continue
            encoded_inputs = tokenizer(token, add_special_tokens=False)
            token_id = encoded_inputs["input_ids"][0]
            if token_id == tokenizer.unk_token_id:
                unk_chars.append(token)
    tokenizer.add_tokens(unk_chars)
    print(f"Added {len(unk_chars)} new tokens")
    return tokenizer


def main():
    parser = HfArgumentParser([TrainingArguments, ExtraTrainingArguments])
    training_args, extra_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    model = AutoModelForTokenClassification.from_pretrained(
        extra_args.model_path, num_labels=4, label2id=LABEL2ID, id2label=ID2LABEL
    )
    tokenizer = AutoTokenizer.from_pretrained(extra_args.model_path)

    if extra_args.dataset_name == "wikipedia":
        wikipedia_dataset = load_from_disk("data/wikipedia-zh-yue/train")
        youtube_df = pd.read_csv("data/youtube_labelled3.csv")
        youtube_dataset = Dataset.from_dict({"text": youtube_df["text"].tolist()})
        train_val_dataset = concatenate_datasets([wikipedia_dataset, youtube_dataset])
    elif extra_args.dataset_name == "youtube":
        youtube_df = pd.read_csv("data/youtube_labelled3.csv")
        train_val_dataset = Dataset.from_dict({"text": youtube_df["text"].tolist()})
    else:
        wikipedia_dataset = load_from_disk("data/wikipedia-zh-yue/train")
        youtube_df = pd.read_csv("data/youtube_labelled3.csv")
        youtube_dataset = Dataset.from_dict({"text": youtube_df["text"].tolist()})
        train_val_dataset = concatenate_datasets([wikipedia_dataset, youtube_dataset])

    if training_args.do_train:
        train_val_dataset_dict = train_val_dataset.train_test_split(0.1)
        train_dataset = train_val_dataset_dict["train"]
        val_dataset = train_val_dataset_dict["test"]

        tokenizer = add_unk_chars(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        train_dataset = train_dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer}, batched=True, num_proc=8)
        val_dataset = val_dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer}, batched=True, num_proc=8)
    else:
        train_dataset = None
        val_dataset = None

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_loss_func=compute_loss_func,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if training_args.do_train:
        trainer.train()

    if training_args.do_predict:
        test_df = pd.read_csv("data/hkcancor.csv")
        test_dataset = Dataset.from_dict({"text": test_df["text"].tolist()})
        test_dataset = test_dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer}, batched=True, num_proc=8)

        prediction_output = trainer.predict(test_dataset)
        trainer.log_metrics("test", prediction_output.metrics)

        preds = []
        for token_ids, pred_ids in zip(test_dataset["input_ids"], prediction_output.predictions):
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            pred = []

            for token, pred_id in zip(tokens[1:-1], pred_ids[1:-1]):
                if pred_id == -100:
                    continue
                pred.append(token)
                if pred_id != 0:
                    pred.append(ID2LABEL[pred_id])

            preds.append("".join(pred))
        pred_df = pd.DataFrame({"text": test_dataset["text"], "pred": preds})
        pred_df.to_csv("outputs/pred.csv", index=False)


if __name__ == "__main__":
    main()
