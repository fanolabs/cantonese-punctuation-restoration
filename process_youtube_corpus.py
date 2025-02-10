import re
from glob import glob

import pandas as pd

from utils import has_chinese, remove_punctuation


def remove_parentheses(text: str) -> str:
    return re.sub("\(.*\)", "", text)


def main():
    texts = []
    for filepath in glob("data/callinter-transcription/*.csv"):
        df = pd.read_csv(filepath)
        df = df.dropna(subset="Transcription")
        for text in df["Transcription"]:
            text = remove_punctuation(text)
            text = remove_parentheses(text)
            if len(text) < 3 or not has_chinese(text):
                continue
            texts.append(text)
    df = pd.DataFrame({"text": texts})
    df.to_csv("data/youtube_unlabelled.csv", index=False)


if __name__ == "__main__":
    main()
