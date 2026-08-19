import  torch
from pathlib import Path

from cs336_alignment.data import tokenize_prompt_and_output
from cs336_alignment.sft import get_response_log_probs



def compute_per_instance_dpo_loss(lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected):
    chosen_response = response_chosen + tokenizer.eos_token
    rejected_response = response_rejected + tokenizer.eos_token
    template_path = (
            Path(__file__).parent
            / "prompts_safety"
            / "alpaca_sft.prompt"
    )
    template = template_path.read_text()
    formatted_prompt = template.split("{response}")[0].format(
        instruction=prompt
    )


    chosen_tokens = tokenize_prompt_and_output(
        prompt_strs=[formatted_prompt],
        output_strs=[chosen_response],
        tokenizer=tokenizer,
    )
    chosen_log_probs = get_response_log_probs(
        model=lm,
        input_ids=chosen_tokens["input_ids"],
        labels=chosen_tokens["labels"],
        return_token_entropy=False,
    )
    chosen_product = chosen_log_probs["log_probs"] * chosen_tokens["response_mask"]
    chosen_log_prob = torch.sum(chosen_product, dim=-1)


    rejected_tokens = tokenize_prompt_and_output(
        prompt_strs=[formatted_prompt],
        output_strs=[rejected_response],
        tokenizer=tokenizer,
    )
    rejected_log_probs = get_response_log_probs(
        model=lm,
        input_ids=rejected_tokens["input_ids"],
        labels=rejected_tokens["labels"],
        return_token_entropy=False,
    )
    rejected_product = rejected_log_probs["log_probs"] * rejected_tokens["response_mask"]
    rejected_log_prob = torch.sum(rejected_product, dim=-1)

    reference_chosen_tokens = tokenize_prompt_and_output(
        prompt_strs=[formatted_prompt],
        output_strs=[chosen_response],
        tokenizer=tokenizer,
    )
    reference_chosen_log_probs = get_response_log_probs(
        model=lm_ref,
        input_ids=chosen_tokens["input_ids"],
        labels=chosen_tokens["labels"],
        return_token_entropy=False,
    )
    reference_chosen_product = reference_chosen_log_probs["log_probs"] * reference_chosen_tokens["response_mask"]
    reference_chosen_log_prob = torch.sum(reference_chosen_product, dim=-1)


    reference_rejected_tokens = tokenize_prompt_and_output(
        prompt_strs=[formatted_prompt],
        output_strs=[rejected_response],
        tokenizer=tokenizer,
    )
    reference_rejected_log_probs = get_response_log_probs(
        model=lm_ref,
        input_ids=rejected_tokens["input_ids"],
        labels=rejected_tokens["labels"],
        return_token_entropy=False,
    )
    reference_rejected_product = reference_rejected_log_probs["log_probs"] * reference_rejected_tokens["response_mask"]
    reference_rejected_log_prob = torch.sum(reference_rejected_product, dim=-1)

    policy_gap = chosen_log_prob - rejected_log_prob
    reference_gap = reference_chosen_log_prob - reference_rejected_log_prob
    logit = beta * (policy_gap - reference_gap)
    loss = -torch.log(torch.sigmoid(logit))
    return loss



