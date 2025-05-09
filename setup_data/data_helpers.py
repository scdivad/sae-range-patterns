import json
from collections import defaultdict, Counter
import torch

def load_data(file_path: str):
    try:
        if "rc_train" in file_path or "key" in file_path or "ioi_train" in file_path or file_path.endswith(".jsonl"):
            with open(file_path, "r") as file:
                data = [json.loads(line) for line in file]
        else:
            with open(file_path, "r") as file:
                data = json.load(file)
        print(f"Loaded Data from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

def process_data_formatted(tokenizer, data, example_length: int, balance_classes=False):
    """
    Filters and balances the dataset based on token length and class counts.
    """
    filtered = [
        entry for entry in data
        if len(tokenizer(entry['prompt']).input_ids) == example_length
    ]
    print(f"Number of filtered entries: {len(filtered)}")
    if not balance_classes:
        prompts = [entry['prompt'] for entry in filtered]
        labels = [entry['label'] for entry in filtered]
        return prompts, labels
    label_counts = Counter(entry['label'] for entry in filtered)
    max_per_label = min(label_counts.values())
    prompts, labels = [], []
    seen = defaultdict(int)
    for entry in filtered:
        label = entry['label']
        if seen[label] < max_per_label:
            prompts.append(entry['prompt'])
            labels.append(label)
            seen[label] += 1
    return prompts, labels

def process_data(tokenizer, data, example_length: int, balance_labels=False):
    """
    Filters and balances the dataset based on token length and class counts.
    """
    # Filter entries where the tokenized clean and patch prefixes both match the expected example length
    filtered_entries = [
        entry for entry in data
        if len(tokenizer(entry['clean_prefix']).input_ids) ==
           len(tokenizer(entry['patch_prefix']).input_ids) == example_length
    ]
    print(f"Number of filtered entries: {len(filtered_entries)}")
    clean_labels_all = [entry['clean_answer'] for entry in filtered_entries]
    corr_labels_all = [entry['patch_answer'] for entry in filtered_entries]
    min_label_counts_clean = min(Counter(clean_labels_all).values()) if balance_labels else float('inf')
    min_label_counts_corr = min(Counter(corr_labels_all).values()) if balance_labels else float('inf')
    running_counts_clean, running_counts_corr = defaultdict(int), defaultdict(int)

    clean_data, corr_data = [], []
    clean_labels, corr_labels = [], []
    for entry in filtered_entries:
        clean_label = entry['clean_answer']
        corr_label = entry['patch_answer']
        if running_counts_clean[clean_label] < min_label_counts_clean and clean_label != "Traceback":
            clean_data.append(entry['clean_prefix'])
            clean_labels.append(clean_label)
            running_counts_clean[clean_label] += 1
        if running_counts_corr[corr_label] < min_label_counts_corr and corr_label != "Traceback":
            corr_data.append(entry['patch_prefix'])
            corr_labels.append(corr_label)
            running_counts_corr[corr_label] += 1

    # print("Final label counts (clean):", Counter(clean_labels))
    # print("Final label counts (corrected):", Counter(corr_labels))

    return (clean_data, corr_data, clean_labels, corr_labels)

def tokenize_and_reshape2(model, clean_data, clean_labels, batch_size: int = 16, max_samples: int = 1e9):
    """
    Tokenizes the input data and reshapes the token tensors into batches.
    """
    total_samples = (min(max_samples, len(clean_data)) // batch_size) * batch_size
    prompt_tokens = model.to_tokens(clean_data[:total_samples])
    prompt_tokens = prompt_tokens.reshape(-1, batch_size, prompt_tokens.shape[-1])
    return prompt_tokens

def tokenize_and_reshape(model, clean_data, corr_data, clean_labels, corr_labels, batch_size: int = 16, max_samples: int = int(1e9)):
    """
    Tokenizes and reshapes clean and corrupted data using tokenize_and_reshape2.
    """
    # Process clean data
    clean_tokens, clean_label_tokens = tokenize_and_reshape2(
        model, clean_data, clean_labels, batch_size, max_samples
    )

    # Process corrupted data
    corr_tokens, corr_label_tokens = tokenize_and_reshape2(
        model, corr_data, corr_labels, batch_size, max_samples
    )

    print("Reshaped token shapes:", clean_tokens.shape, corr_tokens.shape, clean_label_tokens.shape, corr_label_tokens.shape)
    return clean_tokens, corr_tokens, clean_label_tokens, corr_label_tokens

# def tokenize_and_reshape(model, clean_data, corr_data, clean_labels, corr_labels, batch_size: int = 16, max_samples: int = 1e9):
#     """
#     Tokenizes the input data and reshapes the token tensors into batches.
#     """
#     N = min(max_samples, len(clean_data))
#     clean_tokens = model.to_tokens(clean_data[:N])
#     corr_tokens = model.to_tokens(corr_data[:N])
#     clean_label_tokens = model.to_tokens(clean_labels[:N], prepend_bos=False).squeeze(-1)
#     corr_label_tokens = model.to_tokens(corr_labels[:N], prepend_bos=False).squeeze(-1)
#     print("Initial token shapes:", clean_tokens.shape, corr_tokens.shape)

#     # Crop tokens to ensure a multiple of the batch size
#     total_samples = (len(clean_tokens) // batch_size) * batch_size
#     clean_tokens = clean_tokens[:total_samples]
#     corr_tokens = corr_tokens[:total_samples]
#     clean_label_tokens = clean_label_tokens[:total_samples]
#     corr_label_tokens = corr_label_tokens[:total_samples]

#     # Reshape tokens for batching
#     clean_tokens = clean_tokens.reshape(-1, batch_size, clean_tokens.shape[-1])
#     corr_tokens = corr_tokens.reshape(-1, batch_size, corr_tokens.shape[-1])
#     clean_label_tokens = clean_label_tokens.reshape(-1, batch_size)
#     corr_label_tokens = corr_label_tokens.reshape(-1, batch_size)

#     print("Reshaped token shapes:", clean_tokens.shape, corr_tokens.shape, clean_label_tokens.shape, corr_label_tokens.shape)
#     return clean_tokens, corr_tokens, clean_label_tokens, corr_label_tokens

def filter_prompts_by_label(target_c, label_tokens, include_not_target=False):
    if isinstance(target_c, (int)):
        target_tensor = torch.tensor([target_c], device=label_tokens.device)
    else:
        target_tensor = torch.tensor(target_c, device=label_tokens.device)

    mask = torch.isin(label_tokens, target_tensor)

    if include_not_target:
        return mask.nonzero(as_tuple=True)[0], (~mask).nonzero(as_tuple=True)[0]
    else:
        return mask.nonzero(as_tuple=True)[0]

