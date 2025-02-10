import os
import re
from pprint import pprint
from typing import Dict, List

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
)


puncutation_mapping = {
    "O": 0,
    "，": 1,
    "。": 2,
    "？": 3,
}


def split_into_words_and_labels(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+|[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎𠻺𠸏𠺢𡃉𡁵，。？]", text)
    new_words = []
    labels = []
    for word in words:
        if word in puncutation_mapping:
            if labels:
                labels[-1] = puncutation_mapping[word]
        else:
            new_words.append(word)
            labels.append(puncutation_mapping["O"])
    return new_words, labels


# def tokenize_and_align_labels(tokenizer: PreTrainedTokenizer, text: str) -> BatchEncoding:
#     words, labels = split_into_words_and_labels(text)
#     encoded_inputs = tokenizer(words, add_special_tokens=False, is_split_into_words=True)
#     num_tokens = len(encoded_inputs["input_ids"])
#     new_labels = [0] * num_tokens
#     for token_idx in range(num_tokens):
#         word_idx = encoded_inputs.token_to_word(token_idx)
#         new_labels[token_idx] = labels[word_idx]
#     encoded_inputs["labels"] = new_labels
#     return encoded_inputs


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    metrics_dict = {}
    per_label_precisions, per_label_recalls, per_label_f1_scores, _ = precision_recall_fscore_support(
        labels, preds, average=None
    )
    label_names = ["O", "comma", "period", "question"]
    for label_name, precision, recall, f1_score in zip(
        label_names, per_label_precisions, per_label_recalls, per_label_f1_scores
    ):
        if label_name == "O":
            continue
        metrics_dict[f"{label_name}_precision"] = precision
        metrics_dict[f"{label_name}_recall"] = recall
        metrics_dict[f"{label_name}_f1_score"] = f1_score
    macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(labels, preds, average="macro")
    metrics_dict["macro_precision"] = macro_precision
    metrics_dict["macro_recall"] = macro_recall
    metrics_dict["macro_f1_score"] = macro_f1_score
    _, _, micro_f1_score, _ = precision_recall_fscore_support(labels, preds, average="micro")
    metrics_dict["micro_f1_score"] = micro_f1_score
    for k, v in metrics_dict.items():
        metrics_dict[k] = round(v, 4) * 100
    return metrics_dict


def remove_punctuations(text: str) -> str:
    return re.sub("[，。？]", "", text)


def main():
    models = [
        "gpt-35-turbo",
        "gpt-4o-mini",
        "Qwen2.5-72B-Instruct-AWQ",
        "DeepSeek-R1-Distill-Qwen-32B",
        "CantoneseLLMChat-v1.0-7B",
        "Llama-3.3-70B-Instruct-AWQ",
    ]

    # Find a common set of sentences that are valid in all model predictions
    all_indices = set()
    for model in models:
        indices = set()
        count = 0
        df = pd.read_csv(os.path.join("outputs", f"{model}_pred.csv"))
        df = df.fillna("")
        for i, (text, pred) in enumerate(zip(df["text"].tolist(), df["pred"].tolist())):
            text_without_punct = remove_punctuations(text)
            pred_without_punct = remove_punctuations(pred)
            if len(text_without_punct) != len(pred_without_punct):
                count += 1
                continue
            if len(text_without_punct) == len(pred_without_punct) and re.search("[，。？]", pred) is not None:
                indices.add(i)
        print(model, count)

        if all_indices:
            all_indices = all_indices.intersection(indices)
        else:
            all_indices = indices

    print(f"{len(all_indices)}/{len(df)}")

    all_true_labels = []
    all_pred_labels = []
    for model in models:
        df = pd.read_csv(os.path.join("outputs", f"{model}_pred.csv"))
        df = df.dropna()
        for i, (text, pred) in enumerate(zip(df["text"].tolist(), df["pred"].tolist())):
            if i in all_indices:
                _, true_labels = split_into_words_and_labels(text)
                _, pred_labels = split_into_words_and_labels(pred)
                if len(true_labels) == len(pred_labels):
                    all_true_labels += true_labels
                    all_pred_labels += pred_labels
        metrics = compute_metrics(all_true_labels, all_pred_labels)
        metrics["model"] = model
        pprint(metrics, sort_dicts=False)


if __name__ == "__main__":
    main()
