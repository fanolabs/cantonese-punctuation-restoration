import json
import re
from argparse import ArgumentParser
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel
from openai import AzureOpenAI
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from utils import compute_metrics, remove_punctuation


prompt_template = """
# Task Description
Your task is to add appropriate punctuation (commas, full stops, or question marks) to the provided Cantonese text. 
The text is part of a transcript, and the goal is to improve its readability while preserving the original wording.

# Constraints
- Only use commas, full stops, or question marks where they fit naturally based on the context and spoken pauses in Cantonese.
- Do not add, modify, or remove any words from the input text, even if the text is grammatically incorrect or nonsensical.
- Format your response as a JSON object with a single key named "output", so that it can be directly parsed using Python's `json.loads()` without any modifications.
- Do not include any additional text, explanations, or conclusions. Only provide the JSON object.

# Examples
Input: 誒你好我叫Alan姓黃嘅請問有咩幫到你
Output: {{"output": "誒，你好，我叫Alan，姓黃嘅，請問有咩幫到你？"}}

Input: 我聽日要返工今晚要早啲瞓
Output: {{"output": "我聽日要返工，今晚要早啲瞓。"}}

Input: 你食咗飯未呀
Output: {{"output": "你食咗飯未呀？"}}

# Input Text
{text}
"""


puncutation_mapping = {
    "O": 0,
    "，": 1,
    "。": 2,
    "？": 3,
}


class Response(BaseModel):
    output: str


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


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--input_filepath", type=str, default="data/hkcancor.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_filepath).iloc[:10]
    true_punctuated_texts = df["text"].tolist()
    unpunctuated_texts = [remove_punctuation(text) for text in true_punctuated_texts]
    prompts = [prompt_template.format(text=text) for text in unpunctuated_texts]

    pred_punctuated_texts = []
    if args.model.startswith("gpt"):
        client = AzureOpenAI(
            api_version="2024-08-01-preview",
            api_key="API_KEY",
            azure_endpoint="AZURE_ENDPOINT",
        )
        for prompt in tqdm(prompts):
            messages = [{"role": "user", "content": prompt}]
            try:
                response = client.beta.chat.completions.parse(
                    messages=messages,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=1024,
                    response_format=Response,
                )
                pred_punctuated_text = response.choices[0].message.parsed.output
            except:
                pred_punctuated_text = ""
            pred_punctuated_texts.append(pred_punctuated_text)
    else:
        llm = LLM(model=args.model, guided_decoding_backend="outlines")
        json_schema = Response.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=1024,
            guided_decoding=guided_decoding_params,
        )
        outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
        for output in outputs:
            json_str = output.outputs[0].text
            try:
                pred_punctuated_text = json.loads(json_str).get("output", "")
            except:
                pred_punctuated_text = ""
            pred_punctuated_texts.append(pred_punctuated_text)

    all_true_labels = []
    all_pred_labels = []
    for i, (text, pred) in enumerate(zip(true_punctuated_texts, pred_punctuated_texts)):
        _, true_labels = split_into_words_and_labels(text)
        _, pred_labels = split_into_words_and_labels(pred)
        # It is possible that the model generates a text with different length than the original text
        # We only keep the predictions that have the same length as the original text
        if len(true_labels) == len(pred_labels):
            all_true_labels += true_labels
            all_pred_labels += pred_labels
    metrics = compute_metrics(all_true_labels, all_pred_labels)
    print(metrics)


if __name__ == "__main__":
    main()
