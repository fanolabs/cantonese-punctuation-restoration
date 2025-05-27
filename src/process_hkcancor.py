import re
from collections import defaultdict
from typing import List

import pandas as pd
import pycantonese

from utils import has_chinese, remove_punctuation


def format_text(text: str) -> str:
    text = re.sub(r"(?<=[A-Z])\.(?=[A-Z])\.", "", text)
    text = re.sub(r"(?<=[a-zA-Z])-(?=[a-zA-Z])", " ", text)
    text = re.sub(r"(?<![a-zA-Z])-|-(?![a-zA-Z])", "", text)
    text = text.replace("/", "")
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace("„", "，")
    text = text.replace("...", "。")
    text = text.replace("..", "。")
    text = text.replace(".", "。")
    text = text.replace("!", "。")
    text = text.replace(",", "，")
    text = text.replace("?", "？")
    text = re.sub("嘅[0-9]", "嘅", text)
    text = re.sub("到[0-9]", "到", text)
    text = re.sub("噉[0-9]", "噉", text)
    text = text.replace("囖", "囉")
    text = text.replace("噉", "咁")
    text = text.replace("𡃉", "㗎")
    text = text.replace("𠸏", "㗎")
    text = text.replace("𠺢", "㗎")
    text = text.replace("嚹", "喇")
    text = text.replace("嘞", "喇")
    text = text.replace("𠻺", "呀")
    text = text.replace("嗎", "嘛")
    text = text.replace("𠺝", "㗎")
    text = text.replace("𡁵", "緊")
    return text


def join_tokens(tokens: List[str]) -> str:
    """Join a list of Chinese characters and English words into a string. Leave a space between English words."""
    sentence = []
    prev_is_eng = False
    for token in tokens:
        curr_is_eng = re.fullmatch("[a-zA-Z]+", token) is not None
        if prev_is_eng and curr_is_eng:
            sentence.append(" ")
        sentence.append(token)
        prev_is_eng = curr_is_eng
    return "".join(sentence)


def process_corpus(corpus: pycantonese.CHATReader) -> pd.DataFrame:
    seen = defaultdict(list)

    for tokens in corpus.words(by_utterances=True):
        # Skip utterances that contain any Jyutping or special symbol
        if any(re.search("[a-z]+[0-9]", token) or "○" in token for token in tokens):
            continue

        # Skip utterances that do not contain any Chinese character
        if not any(has_chinese(token) for token in tokens):
            continue

        # Standardize punctuation and some Cantonese characters
        tokens = [format_text(token) for token in tokens]

        # Remove empty tokens
        tokens = [token for token in tokens if token]

        # Remove leading punctuation
        while tokens and tokens[0] in "，。？":
            tokens.pop(0)

        # Skip utterances that do not contain any punctuation
        if not any(token in "，。？" for token in tokens):
            continue

        # Join tokens to form a string
        utterance = join_tokens(tokens)
        utterance = utterance.replace("，，", "，").replace("，。", "。").replace("，？", "？").replace("。，", "，")

        # Skip utterances that are too short
        if len(utterance) < 3:
            continue

        utterance_without_punctuations = remove_punctuation(utterance)
        seen[utterance_without_punctuations].append(utterance)

    # Some utterances have the same words but different punctuation, we want to avoid them
    data = []
    for v in seen.values():
        if len(v) == 1:
            data.append({"text": v[0]})
    return pd.DataFrame(data)


def main():
    df = process_corpus(pycantonese.hkcancor())
    df.to_csv("data/hkcancor.csv", index=False)


if __name__ == "__main__":
    main()
