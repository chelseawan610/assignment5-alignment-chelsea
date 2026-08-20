import torch


def compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths):
    raw_reward_values = []
    format_reward_values = []
    answer_reward_values = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_reward_values.append(reward_dict["reward"])
        format_reward_values.append(reward_dict["format_reward"])
        answer_reward_values.append(reward_dict["answer_reward"])
    raw_rewards = torch.tensor(raw_reward_values, dtype=torch.float32)
    format_rewards = torch.tensor(format_reward_values, dtype=torch.float32)
    answer_rewards = torch.tensor(answer_reward_values, dtype=torch.float32)
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": format_rewards.mean().item(),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(raw_rewards, group_size, baseline="mean", advantage_eps=1e-6, advantage_normalizer="std"):
    if raw_rewards.numel() % group_size != 0:
        raise ValueError("raw_rewards must be divisible by group_size")
    groups = raw_rewards.reshape(-1, group_size)
    if baseline == "mean":
        advantages = groups - groups.mean(dim=1, keepdim=True)
    elif baseline == "none":
        advantages = groups.clone()
    else:
        raise ValueError(f"unknown baseline: {baseline}")
    if advantage_normalizer == "std":
        advantages = advantages / (groups.std(dim=1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (groups.mean(dim=1, keepdim=True) + advantage_eps)
    elif advantage_normalizer != "none":
        raise ValueError(f"unknown normalizer: {advantage_normalizer}")
    return advantages.reshape(-1), {"mean_reward": raw_rewards.mean().item()}


def compute_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs, importance_reweighting_method="none", old_log_probs=None, cliprange=None, response_mask=None):
    advantages = raw_rewards_or_advantages.reshape(-1, 1)
    if importance_reweighting_method == "none":
        loss = -(advantages * policy_log_probs)
        return loss, {}
    else:
        if old_log_probs is None:
            raise ValueError("old_log_probs required")
        if importance_reweighting_method == "gspo":
            mask = torch.ones_like(policy_log_probs) if response_mask is None else response_mask
            seq_ratio = torch.exp(((policy_log_probs - old_log_probs) * mask).sum(1) / mask.sum(1).clamp_min(1))
            ratio = seq_ratio[:, None].expand_as(policy_log_probs)
        else:
            ratio = torch.exp(policy_log_probs - old_log_probs)
        if importance_reweighting_method in ("grpo", "gspo"):
            if cliprange is None:
                raise ValueError("cliprange required")
            clipped = ratio.clamp(1 - cliprange, 1 + cliprange)
            loss = -torch.minimum(ratio * advantages, clipped * advantages)
            return loss, {"clip_fraction": (ratio != clipped).float().mean()}
    return -(ratio * advantages), {}


def aggregate_loss_across_microbatch(per_token_policy_gradient_loss, mask, loss_normalization="sequence", normalization_constant=None):
    masked = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        return (masked.sum(1) / mask.sum(1).clamp_min(1)).mean()
    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant required")
        return masked.sum() / normalization_constant
    raise ValueError(f"unknown loss_normalization: {loss_normalization}")


def grpo_train_step(model, tokenizer, optimizer, gradient_accumulation_steps, max_grad_norm, reward_fn, repeated_prompts, rollout_responses, repeated_ground_truths, group_size, baseline="mean", advantage_eps=1e-6, advantage_normalizer="std", importance_reweighting_method="none", old_log_probs=None, cliprange=None, loss_normalization="sequence", normalization_constant=None):
    from .data import tokenize_prompt_and_output
    from .sft import get_response_log_probs
    raw_rewards, metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, reward_meta = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids, labels, response_mask = batch["input_ids"], batch["labels"], batch["response_mask"]
    optimizer.zero_grad()
    micro_losses = []
    loss_meta = {}
    for ids, labs, mask, adv, old in zip(
        input_ids.chunk(gradient_accumulation_steps),
        labels.chunk(gradient_accumulation_steps),
        response_mask.chunk(gradient_accumulation_steps),
        advantages.chunk(gradient_accumulation_steps),
        old_log_probs.chunk(gradient_accumulation_steps) if old_log_probs is not None else [None] * gradient_accumulation_steps,
    ):
        logs = get_response_log_probs(model, ids, labs, False)["log_probs"]
        per_token, loss_meta = compute_policy_gradient_loss(adv, logs, importance_reweighting_method, old, cliprange, mask)
        micro_loss = aggregate_loss_across_microbatch(per_token, mask, loss_normalization, normalization_constant)
        micro_losses.append(micro_loss)
        (micro_loss / gradient_accumulation_steps).backward()
    if loss_normalization == "sequence":
        # The logged batch loss uses token-weighted aggregation; gradients
        # above still follow the requested microbatch normalization.
        with torch.no_grad():
    loss = torch.stack(micro_losses).mean()

    else:
        loss = torch.stack(micro_losses).mean()

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metadata.update(reward_meta); metadata.update(loss_meta); metadata["grad_norm"] = grad_norm
    return loss.detach(), metadata
