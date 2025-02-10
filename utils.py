import json
import re
from typing import Any, Dict


LABEL2ID = {
    "O": 0,
    "，": 1,
    "。": 2,
    "？": 3,
}
ID2LABEL = {label_id: label for label, label_id in LABEL2ID.items()}


def has_chinese(text: str) -> bool:
    """Check if input string contains any Chinese characters."""
    return re.search("[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎𠻺𠸏𠺢𡃉𡁵𠝹𠻹𠵱𨋢𥄫𥚃]", text) is not None


def remove_punctuation(text: str) -> str:
    return re.sub("[，。？]", "", text)


def parse_json_output(text: str) -> Dict[str, Any]:
    opening_bracket_index = text.find("{")
    closing_bracket_index = text.find("}")

    if opening_bracket_index == -1 or closing_bracket_index == -1:
        raise ValueError("JSON parsing error: Can't find an opening curly bracket and/or a closing curly bracket.")

    if opening_bracket_index > closing_bracket_index:
        raise ValueError(
            "JSON parsing error: The first closing curly bracket is found at an earlier index than the first opening curly bracket."
        )

    text = text[opening_bracket_index : closing_bracket_index + 1]
    json_response = json.loads(text)
    return json_response["output"]
