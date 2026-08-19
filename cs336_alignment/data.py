from collections import defaultdict

import torch
from torch import Tensor
import json
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    if len(prompt_strs) != len(output_strs):
        raise ValueError

    sequences = []
    prompt_lengths = []
    total_lengths = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)
        tokens = prompt_tokens + output_tokens
        sequences.append(tokens)
        prompt_lengths.append(len(prompt_tokens))
        total_lengths.append(len(tokens))

    max_length = max(total_lengths)
    batch_size = len(sequences)
    shape = (batch_size, max_length)
    padded_sequences = torch.full(shape, tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros(shape, dtype=torch.long)

    for row_index, sequence in enumerate(sequences):
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        sequence_length = len(sequence)
        padded_sequences[row_index, :sequence_length] = sequence_tensor
        prompt_length = prompt_lengths[row_index]
        total_length = total_lengths[row_index]
        response_mask[row_index, prompt_length:total_length] = 1
    input_ids = padded_sequences[:, :-1]
    labels = padded_sequences[:, 1:]
    response_mask = response_mask[:, 1:]
    return {
  "input_ids": input_ids,
  "labels": labels,
  "response_mask": response_mask,
}

class PackedSFTDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return self.examples[index]

def run_get_packed_sft_dataset(tokenizer, dataset_path, seq_length, shuffle):
    template_path = Path(__file__).parent / "prompts_safety" / "alpaca_sft.prompt"
    template = template_path.read_text().rstrip()

    documents = []
    with open(dataset_path) as f:
        for line in f:
            text = json.loads(line)
            documents.append(text)
    if shuffle:
        random.shuffle(documents)

    token_stream = []

    for document in documents:
        text = template.format(instruction=document["prompt"],
    response=document["response"],)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens.append(tokenizer.eos_token_id)
        token_stream.extend(tokens)

    examples = []
    chunk_size = seq_length + 1
    for start in range(0, len(token_stream)-seq_length, seq_length):
        chunk = token_stream[start : start + chunk_size]
        input_ids = chunk[:-1]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = chunk[1:]
        labels = torch.tensor(labels, dtype=torch.long)
        dictionary = {
    "input_ids": input_ids,
    "labels": labels,
}
        examples.append(dictionary)
    return PackedSFTDataset(examples)

def iterate_batches(dataset, batch_size, shuffle):
    dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
)
    return dataloader








