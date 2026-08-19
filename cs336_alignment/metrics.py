import re


def parse_mmlu_response(mmlu_example, model_output):
    normalized_output = model_output.upper()
    match = re.search(pattern = r"CORRECT ANSWER IS\s*([A-D])", string = normalized_output)
    if match:
        return match.group(1)
    else:
        return None





def parse_gsm8k_response(model_output):
    matches = re.findall(pattern=r"\d+", string=model_output)
    if matches:
        return matches[-1]
    else:
        return None
