import re
from typing import Dict, List

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def has_chinese(text: str) -> bool:
    """Check if input string contains any Chinese characters."""
    match = re.search("[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎𠻺𠸏𠺢𡃉𡁵𠝹𠻹𠵱𨋢𥄫𥚃]", text)
    return match is not None


def remove_punctuation(text: str) -> str:
    return re.sub("[，。？]", "", text)


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    # Initialize dictionaries to store results
    label_names = ["comma", "period", "question"]
    metrics_dict = {}
    total_tn = total_fp = total_fn = total_tp = 0

    # Calculate precision, recall, and F1 for each class
    for i, label_name in enumerate(label_names, start=1):
        # Convert predictions and labels to binary for the current class
        preds_binary = [1 if pred == i else 0 for pred in preds]
        labels_binary = [1 if label == i else 0 for label in labels]

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(labels_binary, preds_binary).ravel()
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp
        precision = precision_score(labels_binary, preds_binary, zero_division=0)
        recall = recall_score(labels_binary, preds_binary, zero_division=0)
        f1 = f1_score(labels_binary, preds_binary, zero_division=0)

        # Store results
        metrics_dict[f"{label_name}_p"] = precision
        metrics_dict[f"{label_name}_r"] = recall
        metrics_dict[f"{label_name}_f"] = f1

    # Calculate macro averages
    metrics_dict["macro_p"] = sum(metrics_dict[f"{label_name}_p"] for label_name in label_names) / len(label_names)
    metrics_dict["macro_r"] = sum(metrics_dict[f"{label_name}_r"] for label_name in label_names) / len(label_names)
    metrics_dict["macro_f"] = sum(metrics_dict[f"{label_name}_f"] for label_name in label_names) / len(label_names)
    metrics_dict["micro_p"] = total_tp / (total_tp + total_fp)
    metrics_dict["micro_r"] = total_tp / (total_tp + total_fn)
    metrics_dict["micro_f"] = (
        2 * metrics_dict["micro_p"] * metrics_dict["micro_r"] / (metrics_dict["micro_p"] + metrics_dict["micro_r"])
    )

    for k, v in metrics_dict.items():
        metrics_dict[k] = round(v * 100, 1)
    return metrics_dict
