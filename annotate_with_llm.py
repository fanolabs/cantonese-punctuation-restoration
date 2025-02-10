import os
from argparse import ArgumentParser
from functools import partial
from typing import Callable, Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import parse_json_output, remove_punctuation


load_dotenv()


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


def openai_generate(
    all_messages: List[List[Dict[str, str]]],
    texts: List[str],
    client: AzureOpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
) -> List[str]:
    outputs = []
    for messages, text in tqdm(zip(all_messages, texts), total=len(texts)):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["<|im_end|>"],
            )
            output = response.choices[0].message.content
            if output is None:
                output = f'{{"output": "{text}"}}'
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            output = f'{{"output": "{text}"}}'
        outputs.append(output)
    return outputs


def vllm_generate(
    all_messages: List[List[Dict[str, str]]], texts: List[str], llm: LLM, sampling_params: SamplingParams
) -> List[str]:
    responses = llm.chat(all_messages, sampling_params)
    return [response.outputs[0].text for response in responses]


def annotate(
    texts: List[str],
    send_to_llm: Callable[[List[List[Dict[str, str]]]], List[str]],
    max_retries: int = 3,
) -> List[str]:
    outputs = [None] * len(texts)
    all_messages = [[{"role": "user", "content": prompt_template.format(text=text)}] for text in texts]

    retry_count = 0
    while retry_count < max_retries:
        # Identify prompts that still need processing
        remaining_messages = []
        remaining_texts = []
        remaining_indices = []
        for i, output in enumerate(outputs):
            if output is None:  # Only process prompts without valid outputs
                remaining_messages.append(all_messages[i])
                remaining_texts.append(texts[i])
                remaining_indices.append(i)

        # If all outputs are valid, break the loop
        if not remaining_messages:
            break

        # Send remaining prompts to the LLM
        llm_outputs = send_to_llm(remaining_messages, remaining_texts)

        # Verify each output and update outputs
        for i, llm_output in zip(remaining_indices, llm_outputs):
            try:
                llm_output = llm_output.replace("！", "。")
                # llm_output = re.sub(r"<think>.*</think>", "", llm_output)
                parsed_llm_output = parse_json_output(llm_output)
                parsed_llm_output_without_punct = remove_punctuation(parsed_llm_output)
                if parsed_llm_output_without_punct != texts[i]:
                    raise ValueError(
                        "The output must contain the exact same words as the input. \n"
                        f"The input is: {texts[i]}\n"
                        f"But your output (without punctuation) is: {parsed_llm_output_without_punct}\n"
                        "Do not add, modify, or remove any words from the input text. "
                    )
                outputs[i] = parsed_llm_output
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(e)
                print(llm_output)
                print("------------")
                # Append the error message to the prompt for the next retry
                all_messages[i].append({"role": "assistant", "content": llm_output})
                all_messages[i].append(
                    {
                        "role": "user",
                        "content": (
                            f"{e}\nPlease fix the previous response. "
                            "Format your response in a JSON object as mentioned in the task description. "
                            "Do not include any additional text, explanations, or conclusions. Only provide the JSON object.",
                        ),
                    }
                )

        retry_count += 1

    return outputs


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    input_df = pd.read_csv(args.data_filepath)
    texts = input_df["text"].tolist()
    texts_without_punctuation = input_df["text"].map(remove_punctuation).tolist()

    if args.model in ("gpt-35-turbo", "gpt-4o-mini", "gpt-4o"):
        client = AzureOpenAI(
            api_version="2023-12-01-preview",
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
        )
        send_to_llm = partial(
            openai_generate, model=args.model, client=client, temperature=args.temperature, max_tokens=args.max_tokens
        )
    else:
        llm = LLM(
            model=os.path.join("/mnt/nas2/Pretrained_LM", args.model),
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=False,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
        )
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
        send_to_llm = partial(vllm_generate, llm=llm, sampling_params=sampling_params)

    outputs = annotate(texts_without_punctuation, send_to_llm, max_retries=args.max_retries)

    output_df = pd.DataFrame({"text": texts, "pred": outputs})
    output_df.to_csv(f"outputs/{args.model}_pred.csv", index=False)


if __name__ == "__main__":
    main()
