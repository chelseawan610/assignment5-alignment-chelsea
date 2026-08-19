import torch
import torch.nn.functional as F


def get_response_log_probs(model, input_ids, labels, return_token_entropy):
    outputs = model(input_ids = input_ids)
    logits = outputs.logits

    all_log_probs = F.log_softmax(logits, dim=-1)
    label_indices = labels.unsqueeze(-1)
    log_prob = torch.gather(input=all_log_probs, dim=-1, index=label_indices)

    log_probs = log_prob.squeeze(-1)
    result = {"log_probs" : log_probs}
    if return_token_entropy:
        probs = torch.exp(all_log_probs)
        token_entropy = -torch.sum(probs * all_log_probs, dim=-1)
        result["token_entropy"] = token_entropy
    return result




