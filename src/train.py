import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
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

from utils import compute_metrics, has_chinese


os.environ["TOKENIZERS_PARALLELISM"] = "false"


LABEL2ID = {"O": 0, "，": 1, "。": 2, "？": 3}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


@dataclass
class ExtraTrainingArguments:
    model_name_or_path: str
    train_dataset: Optional[str] = None
    test_dataset: Optional[str] = None
    project_name: Optional[str] = None


def load_data(filepath: str) -> Dataset:
    datasets = []
    for path in filepath.split(","):
        if path.endswith("csv"):
            dataset = load_dataset("csv", data_files=path, split="train")
        else:
            dataset = load_from_disk(path)
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def preprocess_logits_for_metrics(logits, labels) -> torch.Tensor:
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics_wrapper(eval_preds: EvalPrediction) -> Dict[str, float]:
    preds = []
    labels = []
    for pred, label in zip(eval_preds.predictions.flatten(), eval_preds.label_ids.flatten()):
        if label != -100:
            preds.append(pred)
            labels.append(label)

    return compute_metrics(labels, preds)


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

    encoded_inputs = tokenizer(batch_words_without_punct, is_split_into_words=True, truncation=True, max_length=512)

    batch_size = len(encoded_inputs["input_ids"])
    batch_aligned_labels = []
    for batch_idx in range(batch_size):
        seq_len = len(encoded_inputs["input_ids"][batch_idx])
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
    token_list_filepath = "data/asr_vocab.txt"
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
        extra_args.model_name_or_path, num_labels=4, label2id=LABEL2ID, id2label=ID2LABEL
    )
    tokenizer = AutoTokenizer.from_pretrained(extra_args.model_name_or_path)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    if training_args.do_train:
        tokenizer = add_unk_chars(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        train_val_dataset = load_data(extra_args.train_dataset)
        train_val_dataset = train_val_dataset.map(
            tokenize, fn_kwargs={"tokenizer": tokenizer}, batched=True, num_proc=8
        )
        train_val_dataset_dict = train_val_dataset.train_test_split(0.1)
        train_dataset = train_val_dataset_dict["train"]
        val_dataset = train_val_dataset_dict["test"]
    else:
        train_dataset = None
        val_dataset = None

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_wrapper,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if "wandb" in training_args.report_to:
        wandb.init(
            config=training_args.__dict__,
            project=extra_args.project_name,
            name=training_args.run_name,
        )

    if training_args.do_train:
        trainer.train()

    if training_args.do_predict:
        test_dataset = load_data(extra_args.test_dataset)
        test_dataset = test_dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer}, batched=True, num_proc=8)

        prediction_output = trainer.predict(test_dataset)

        if "wandb" in training_args.report_to:
            wandb.log(prediction_output.metrics)

        print(f"Model: {extra_args.model_name_or_path}")
        print(f"Train dataset: {extra_args.train_dataset}")
        print(f"Test dataset: {extra_args.test_dataset}")

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
        pred_df.to_csv(f"{training_args.output_dir}/pred.csv", index=False)

    if "wandb" in training_args.report_to:
        wandb.finish()


if __name__ == "__main__":
    main()
