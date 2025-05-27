import re
from typing import Any, Dict, List

from datasets import load_dataset


def split_into_sentences(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    new_examples = {"text": []}
    for text in examples["text"]:
        text = re.sub(r"[【】「」（）《》]", "", text)
        text = text.replace("：", "，")
        text = text.replace("、", "，")
        text = text.replace("；", "，")
        text = text.replace("！", "。")
        parts = re.split(r"(?<=[。？])", text)
        for part in parts:
            if not re.search("[，。？]", text):
                continue
            m = re.search(
                "[^0-9a-zA-Z⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎𠻺𠸏𠺢𡃉𡁵𠝹𠻹𠵱𨋢𥄫𥚃，。？]", part
            )
            if not m:
                part = part.strip()
                if part and len(part) >= 3:
                    new_examples["text"].append(part)
    return new_examples


def main():
    dataset = load_dataset("indiejoseph/wikipedia-zh-yue-filtered")
    dataset = dataset.remove_columns(["title"])
    dataset = dataset.map(split_into_sentences, batched=True)
    dataset.save_to_disk("data/wikipedia-zh-yue")


if __name__ == "__main__":
    main()
