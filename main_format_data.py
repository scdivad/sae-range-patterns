# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from transformers import AutoTokenizer
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import itertools
from torch.serialization import add_safe_globals
from typing import NamedTuple, Iterable, Mapping, Sequence, Union

from setup_data.sparse_matrix import SparseMatrix
from setup_data.model_helpers import (
    load_model_and_saes, 
    model_run_activations, 
    evaluate_on_task, 
    get_latent_explanation,
)
from setup_data.data_helpers import load_data, process_data_formatted, tokenize_and_reshape2, filter_prompts_by_label


class LatentRangeEntry(NamedTuple):
    latent_idx: int
    sae_name: str
    lb: float
    ub: float
    kl: float
    p_y: np.ndarray # P(Y=y | lb <= latent <= ub)
    p_range: np.ndarray # P(lb <= latent <= ub | Y=y)
    p_y_max: float
    p_range_max: float
    pred_class: int

def get_all_activations(model, saes, sae_names, prompt_dataset):
    """
    Returns a sparse version of all SAE activations of the last position over prompt_dataset forward passes.
    The full SAE activation can be retrieved via all_acts_train[sae_name].to_dense() of (n_samples, d_SAE)
    """
    all_acts_train = {sae_name: [] for sae_name in sae_names}
    for batch in tqdm(prompt_dataset, desc="Running model on dataset"):
        batch_acts = model_run_activations(model, saes, batch)
        for sae_name, sae_act in batch_acts.items():
            all_acts_train[sae_name].append(sae_act)
    for sae_name in all_acts_train:
        all_acts_train[sae_name] = SparseMatrix(torch.cat(all_acts_train[sae_name], dim=0))
    return all_acts_train

# %%
def build_important_latents(all_entries, sae_names, classes, score):
    '''
    entries: list of LatentRangeEntry for every single latent and range
    for every single latent

    Returns
    -------
    important_latents : dict[str, dict[int, list[tuple]]]
        SAE layer name ➜ {class_id: [(latent_idx, lb, ub), …], …}.
    '''

    important_latents = {}
    for s in sae_names:
        important_latents[s] = {}
        for c in classes:
            important_latents[s][c] = {} # when returning, this will be type list
            # ... bad coding practice

    for e in all_entries:
        s, c = e.sae_name, e.pred_class
        if e.p_range_max > 0.7 and e.p_y_max > 0.7:
            if e.latent_idx not in important_latents[s][c] or score(e) > score(important_latents[s][c][e.latent_idx]):
                important_latents[s][c][e.latent_idx] = e

    for s in sae_names:
        for c in classes:
            if not important_latents[s][c]:
                important_latents[s][c] = [(0, -5, -5)]
            else:
                important_latents[s][c] = [(e.latent_idx, e.lb, e.ub) for e in important_latents[s][c].values()]

    print(important_latents)
    return important_latents
# %%
def get_important_training_activations(
        all_acts_train, 
        important_latent_idx_ranges, 
        label_dataset,
):
    """
    Parameters
    ----------
    all_acts_train : dict[str, torch.Tensor (n_samples, d_SAE)] see get_all_activations
    important_important_latent_idx_rangeslatent_idxs : dict[str, dict[str, list[tuple]]]
        SAE layer name ➜ {class_id: [(latent_idx, lb, ub), …], …}.
        Only the first element (latent_idx) is used here; lb/ub are ignored.

    Returns
    -------
    dict[str, dict[str, torch.Tensor]]
        SAE layer name ➜ {class_id: (n_samples, d_SAE_nnz)} comprsesed SAE activations such that
        important_training_acts_{s,c}[:,i] of (n_samples,) is the latent activations for important_latents_{s,c}[i]
    """
    important_training_acts = {}
    for s in important_latent_idx_ranges:
        important_training_acts[s] = {}
        for c in important_latent_idx_ranges[s]:
            prompt_idx_target = filter_prompts_by_label(c, label_dataset, include_not_target=False).unsqueeze(1)
            idxs = torch.tensor([tup[0] for tup in important_latent_idx_ranges[s][c]])
            tmp = all_acts_train[s].to_dense()
            important_training_acts[s][c] = tmp[prompt_idx_target, idxs]
            del tmp
    return important_training_acts
# # %%
def scoring_all_ranges(
    latent_idx: int,
    sae_name: str,
    neuron_vals: np.ndarray,
    labels: Sequence[str],
    criteria=None,
    nbins: int = 100,
    min_samples: int = 10,
):
    classes_tokens = np.unique(labels)
    total_counts   = np.bincount(labels,
                                 minlength=classes_tokens.size).astype(np.float32)

    data      = []
    linspace  = np.linspace(neuron_vals.min(), neuron_vals.max(), nbins)

    for i, lb in enumerate(linspace):
        for ub in linspace[i + 1:]:
            in_range = (neuron_vals >= lb) & (neuron_vals <= ub)
            range_lbls = labels[in_range]
            if range_lbls.size < min_samples:
                continue

            counts   = np.bincount(range_lbls,
                                   minlength=classes_tokens.size)
            p_y      = counts / counts.sum()
            p_range  = counts / total_counts
            kl       = np.sum(p_y[p_y > 0] * np.log(p_y[p_y > 0] *
                                                    len(classes_tokens)))

            entry = LatentRangeEntry(
                latent_idx=latent_idx,
                sae_name=sae_name,
                lb=lb,
                ub=ub,
                kl=kl,
                p_y=p_y.copy(),
                p_range=p_range.copy(),
                p_y_max=p_y.max(),
                p_range_max=p_range.max(),
                pred_class=classes_tokens[p_y.argmax()],
            )
            if entry.kl > 0.1 and entry.p_y_max > 0.1 and entry.p_range_max > 0.1:
                data.append(entry)
    return data
# %%
def scoring_all_ranges_fast(
    latent_idx: int,
    sae_name: str,
    x: np.ndarray, # neuron vals, TODO: rename to v
    y: np.ndarray, # labels
    criteria=None,
    nbins: int = 10,
    min_samples: int = 10,
):
    """
    latent_idx: j
    x: (N,) v^{(i)}_j particular SAE latent value for all seen examples
    y: (N,) y^{(i)} of all seen examples

    I will omit j in v_j since we only for the remainder of comments in this function
    """
    N = x.shape[0]
    classes = np.unique(y)
    n_cls = classes.size
    total_counts = np.bincount(y, minlength=n_cls)

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Populates psum of shape (n_class, N + 1) where
    # psum[c][i] gives the count of class c from position 0 to i-1 of y
    psum = np.zeros((n_cls, N + 1), dtype=int)

    # This np.add.at operation is the same as:
    # for i in range(N):
    #   position = i + 1
    #   psum[y[i], position] += 1
    np.add.at(psum, (y, np.arange(N) + 1), 1)

    psum = np.cumsum(psum, axis=1)

    # This triu_indices operation is the same as
    # lb_ids, ub_ids = [], []
    # for i in range(nbins):
    #   for j in range(i+1, nbins):
    #       lb_ids.append(i); ub_ids.append(j)
    # lb_ids = np.array(lb_ids); ub_ids = np.array(ub_ids)
    lb_ids, ub_ids = np.triu_indices(nbins, k=1)

    # Assume v is sorted (it is)
    # For all lb in linspace, ub in linspace, we want the count of lb<=v<=ub and label=c over all N samples.
    # That is the same as the number of positions between l,r inclusive, 0<=l,r<N, where
    # r is the last idx of v over N samples such that v[idx] <= ub.
    # l is the first idx of v over N samples such that lb <= v[idx].
    # Had psum[c][i] given us the count from position 0 to i inclusive then the count would be
    # psum[c][r] - psum[c][l-1], with psum[c][-1]:=0. 
    # Since we are one off to remove edge case handling, it should be psum[c][r+1] - psum[c][l].
    # So, let ub_pos=r+1, the first idx of v that satisfies ub < v[idx], i.e. v[idx-1] <= ub < v[idx]
    # and lb_pos=l, the first idx of v that satisfies lb <= v[idx], i.e. v[idx-1] < lb <= v[idx]
    # https://numpy.org/doc/2.2/reference/generated/numpy.searchsorted.html
    linspace = np.linspace(x.min(), x.max(), nbins)
    lb_pos = np.searchsorted(x, linspace, side='left') 
    ub_pos = np.searchsorted(x, linspace, side='right')
    # instead of grid search, better to make intervals
    # that model the distribution of x
    counts_in_lb = psum[:, lb_pos[lb_ids]]
    counts_in_ub = psum[:, ub_pos[ub_ids]]
    counts = (counts_in_ub - counts_in_lb).astype(float)
    n_in_range = counts.sum(axis=0)

    valid_mask = n_in_range >= min_samples
    if not np.any(valid_mask):
        return []

    counts = counts[:, valid_mask]
    n_in_range = n_in_range[valid_mask]
    lb_ids = lb_ids[valid_mask]
    ub_ids = ub_ids[valid_mask]

    p_y = counts / n_in_range
    log_term = np.where(p_y > 0, np.log(p_y * n_cls), np.zeros_like(p_y))
    kl = (p_y * log_term).sum(axis=0)

    p_range = counts / total_counts[:, np.newaxis]
    p_y_max = np.max(p_y, axis=0)
    p_range_max = np.max(p_range, axis=0)

    select_mask = (kl > 0.1) & (p_y_max > 0.1) & (p_range_max > 0.1)

    if not np.any(select_mask):
        return []

    sel_idx = np.nonzero(select_mask)[0]
    pred_cls_idx = np.argmax(p_y[:, sel_idx], axis=0)

    data = []
    for k, idx in enumerate(sel_idx):
        entry = LatentRangeEntry(
            latent_idx=latent_idx,
            sae_name=sae_name,
            lb=linspace[lb_ids[idx]],
            ub=linspace[ub_ids[idx]],
            kl=kl[idx],
            p_y=p_y[:, idx].copy(),
            p_range=p_range[:, idx].copy(),
            p_y_max=p_y_max[idx],
            p_range_max=p_range_max[idx],
            pred_class=classes[pred_cls_idx[k]],
        )
        data.append(entry)

    return data
# %%

def make_rules(
    important_latent_idx_ranges: dict[str, dict[str, list[tuple]]],
    important_training_acts: dict[str, dict[str, torch.Tensor]],
    k: int = 1,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Generate up to top-k rule patterns per class based on activation intervals.

    Args:
        important_latent_idx_ranges: mapping from layer to class to list of (latent_idx, lb, ub).
        important_training_acts: mapping from layer to class to activations tensor of shape (N, num_latents).
        k: number of top patterns (rules) to keep per class.

    Returns:
        rules_all_layers: mapping from layer to class to Tensor of shape (k, num_entries) containing boolean rule masks.
    """
    rules_all_layers: dict[str, dict[str, torch.Tensor]] = {}
    for layer, class_ranges in important_latent_idx_ranges.items():
        rules_all_layers[layer] = {}
        for cls, entries in class_ranges.items():
            device = important_training_acts[layer][cls].device
            latent_idxs = [tup[0] for tup in entries]
            map_idx_to_col = {latent_idx: col for col, latent_idx in enumerate(latent_idxs)}

            intervals = torch.tensor([[lb, ub] for (_, lb, ub) in entries], device=device)
            col_sel = torch.tensor([map_idx_to_col[i] for i in latent_idxs], dtype=torch.long, device=device)
            act_target = important_training_acts[layer][cls][:, col_sel]

            # TODO: changed to <= =>
            rule_mask = (act_target >= intervals[:, 0]) & (act_target <= intervals[:, 1])

            patterns, cnts = torch.unique(rule_mask, dim=0, return_counts=True)
            print(cls, patterns, cnts)

            k_eff = min(k, cnts.size(0))
            topk_cnts, topk_idxs = torch.topk(cnts, k_eff, largest=True)
            topk_patterns = patterns[topk_idxs]

            rules_all_layers[layer][cls] = topk_patterns

    return rules_all_layers
# %%
def get_prediction(
    activation: torch.Tensor,
    rules_by_class: dict[str, torch.Tensor],
    sae_layer_for_rules: str,
    important_latent_idx_ranges: dict[str, dict[int, list[tuple]]],
) -> list:
    """
    Predict classes whose any of the top-k rules match the given activation.

    Args:
        activation: Tensor of activations for all latents at a given layer.
        rules_by_class: mapping from class to Tensor of shape (k, num_entries) of boolean masks.
        sae_layer_for_rules: the layer key to use in `important_latent_idx_ranges`.
        important_latent_idx_ranges: mapping from layer to class to entries for reconstructing active_set.
        prompt: (unused) original prompt.
        gt: (unused) ground truth label.

    Returns:
        List of classes c for which at least one rule matches.
    """
    layer = sae_layer_for_rules
    predictions = []
    for cls, patterns in rules_by_class.items():
        entries = important_latent_idx_ranges[layer][cls]
        active_set = torch.tensor([
            bool(lb <= activation[idx].item() <= ub)
            for (idx, lb, ub) in entries
        ], dtype=torch.bool)
        for rule_mask in patterns:
            # which is better: if torch.all(rule_mask == active_set)
            # or if torch.all(active_set[rule_mask])?
            if torch.all(rule_mask == active_set):
                predictions.append(cls)
                break
    return predictions

def apply_rule_mask_to_sets_with_ranges(
    rules_all_layers: dict[str, dict[str, torch.Tensor]],
    important_latent_idx_ranges: dict[str, dict[str, list[tuple[int, float, float]]]],
) -> dict[str, dict[str, list[list[tuple[int, tuple[float, float]]]]]]:
    """
    Converts rule masks to lists of (latent index, activation range) for each rule.

    Args:
        rules_all_layers: output of `make_rules`, maps layer -> class -> tensor of rule masks.
        important_latent_idx_ranges: maps layer -> class -> list of (latent_idx, lb, ub).

    Returns:
        rules_as_detailed: maps layer -> class -> list of rules, where each rule is a list of
                           (latent_idx, (lb, ub)) for active indices.
    """
    rules_as_detailed = {}
    for layer, class_rules in rules_all_layers.items():
        rules_as_detailed[layer] = {}
        for cls, rule_masks in class_rules.items():
            latent_info = important_latent_idx_ranges[layer][cls]  # List of (latent_idx, lb, ub)
            detailed_rules = []
            for rule_mask in rule_masks:
                active_features = []
                for active, (latent_idx, lb, ub) in zip(rule_mask.tolist(), latent_info):
                    if active:
                        active_features.append((int(latent_idx), (float(lb), float(ub))))
                detailed_rules.append(active_features)
            rules_as_detailed[layer][cls] = detailed_rules
    return rules_as_detailed
# %%
def main():
    pass # so running this cell doesn't error in interactive mode
    # %%
    add_safe_globals([SparseMatrix])

    hf_cache = "/home/davidsc2/"

    on_delta = False
    if on_delta:
        print("ON DELTA")
        hf_cache = "/u/dcheung2/content/hf_cache"

    os.environ["HF_HOME"] = hf_cache
    print(f"Using Hugging Face cache directory: {hf_cache}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # %% 
    # INPUT PARAMS
    model_name = "google/gemma-2-2b"
    sae_layer_idx = 23

    # for neuronpedia annotations
    model_id = "gemma-2-2b"
    expl_layer = f"{sae_layer_idx}-gemmascope-res-16k"

    task = "key"
    if task == "sva":
        batch_size = 16
        max_samples_train = 6000
        max_samples_test = 1000
        acts_train_path = f'format_sva_train.pt'
        acts_test_path = f'format_sva_test.pt'
        file_path = "format_sfc/formatted_rc_train.json"
        example_length = 7
        topk_rules = 2
    elif task == "ioi":
        batch_size = 16
        max_samples_train = 4992 # fix invisible by 16 bug somewhere
        max_samples_test = 992
        acts_train_path = f'acts_format_ioi_train.pt'
        acts_test_path = f'acts_format_ioi_test.pt'
        file_path = "format_sfc/formatted_ioi_train.json"
        example_length = 20
        topk_rules = 2
    elif task == "key":
        batch_size = 16
        max_samples_train = 6000
        max_samples_test = 992
        acts_train_path = f'train_acts_key_no_err_resid_{sae_layer_idx}_max_samples_{max_samples_train}.pt'
        acts_test_path = f'test_acts_key_no_err_resid_{sae_layer_idx}_max_samples_{max_samples_test}.pt'
        acts_train_path = f'all_training_activations_key_no_err_resid_{sae_layer_idx}_max_samples_{max_samples_train}.pt'
        acts_train_path = f'all_training_activations_key_no_err_resid_{sae_layer_idx}_max_samples_{max_samples_test}.pt'
        file_path = "format_sfc/formatted_key_no_traceback.json"
        example_length = 41
        topk_rules = 1

    # %%
    if not (os.path.exists(acts_train_path) and os.path.exists(acts_test_path)):
        model, saes = load_model_and_saes(model_name, [sae_layer_idx], device, hf_cache)
    else:
        print("Not loading model and saes, using precomputed activations")
        model, saes = None, None
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # %% prepare and process data
    data = load_data(file_path)
    assert data is not None
    print("First Entry:", data[0] if isinstance(data, list) else data)

    clean_prompts, clean_labels = process_data_formatted(tokenizer, data, example_length)
    label_dataset = clean_labels[:max_samples_train]
    
    # %%
    # sae_names = [sae.cfg.hook_name for sae in saes]
    sae_names = [f'blocks.{sae_layer_idx}.hook_resid_post']
    if os.path.exists(acts_train_path):
        all_acts_train = torch.load(acts_train_path)
        print("loading activations from", acts_train_path)
    else:
        clean_prompt_tokens = tokenize_and_reshape2(
            model, 
            clean_prompts, clean_labels, 
            batch_size=batch_size, max_samples=max_samples_train
        )
        prompt_dataset = clean_prompt_tokens
        all_acts_train = get_all_activations(model, saes, sae_names, prompt_dataset)
        torch.save(all_acts_train, acts_train_path)
        print("training activations saved to", acts_train_path)

    nbins = 10               # Number of candidate bins for activation intervals
    min_samples = 10         # Minimal examples needed for a valid interval

    label_dataset_flat_np = np.array(label_dataset)

    # %%
    entries = []
    for sae_name in sae_names:
        dense_acts = all_acts_train[sae_name].to_dense()
        active_mask = (dense_acts > 0).any(dim=0)
        active_indices = torch.nonzero(active_mask).flatten()
        print(f"Layer {sae_name}: {len(active_indices)} out of {dense_acts.shape[1]} neurons are active (> 0) at least once.")
        for i in tqdm(active_indices.tolist(), desc=f"Processing {sae_name} neurons"):
            neuron_vals = dense_acts[:, i]
            entries += scoring_all_ranges_fast(
                i,
                sae_name,
                neuron_vals,
                label_dataset_flat_np,
                criteria=lambda entry: entry.kl > 0.1 and entry.p_y_max > 0.1 and entry.p_range_max > 0.1,
                nbins=nbins, 
                min_samples=min_samples,
            )
        
        del dense_acts
    
    print(len(entries))
    # %%
    # entries = []
    # for sae_name in sae_names:
    #     dense_acts = all_acts_train[sae_name].to_dense().cpu().numpy()
    #     active_mask = (dense_acts > 0).any(axis=0)
    #     active_indices = list(active_mask.nonzero()[0])
    #     print(f"Layer {sae_name}: {len(active_indices)} out of {dense_acts.shape[1]} neurons are active (> 0) at least once.")

    #     def _score_neuron(i):
    #         neuron_vals = torch.from_numpy(dense_acts[:, i])
    #         return scoring_all_ranges(
    #             i,
    #             sae_name,
    #             neuron_vals,
    #             label_dataset_flat_np,
    #             criteria=lambda entry:  entry.p_y_max > 0.1
    #                                     and entry.p_range_max > 0.1,
    #             nbins=nbins,
    #             min_samples=min_samples,
    #         )

    #     results = process_map(
    #         _score_neuron,
    #         active_indices,
    #         desc=f"Processing {sae_name} neurons",
    #         max_workers=None,
    #         chunksize=1
    #     )

    #     entries.extend(itertools.chain.from_iterable(results))
    #     del dense_acts

    # print(len(entries))
    # %%
    # latent activation statistics
    i = 0
    num_plot = 10
    entries.sort(key=lambda x: -x.p_range_max)
    for entry in entries:
        class_tokenized = entry.pred_class
        # class_cleantext = tokenizer.decode(torch.tensor(entry.pred_class))
        if entry.p_y_max > 0.7 and entry.p_range_max > 0.7:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(
                f"Class {class_tokenized} {class_tokenized} | "
                f"idx={entry.latent_idx}, lb={entry.lb:.2f}, ub={entry.ub:.2f}, "
                f" kl={entry.kl:.4f}"
                f" p_y_max={entry.p_y_max:.4f}"
                f" p_range_max={entry.p_range_max:.4f}"
            )

            # P(Y=y | lb ≤ x ≤ ub)
            axes[0].bar(range(len(entry.p_y)), entry.p_y)
            axes[0].set_title('P(Y=y | lb ≤ x ≤ ub)')
            axes[0].set_xlabel('Class y')
            axes[0].set_ylabel('Probability')
            axes[0].set_ylim(0, 1.0)

            # P(lb ≤ x ≤ ub | Y=y)
            axes[1].bar(range(len(entry.p_range)), entry.p_range)
            axes[1].set_title('P(lb ≤ x ≤ ub | Y=y)')
            axes[1].set_xlabel('Class y')
            axes[1].set_ylabel('Probability')
            axes[1].set_ylim(0, 1.0)

            plt.tight_layout()
            plt.show()
            i += 1
            if i == num_plot:
                break
    # %%
    important_latent_idx_ranges = build_important_latents(
        entries,
        sae_names,
        classes=[i for i in range(len(np.unique(clean_labels)))],
        score=lambda entry: (math.log(entry.p_y_max) + math.log(entry.p_range_max), -(entry.ub - entry.lb))
    )
    # %%

    important_training_acts = get_important_training_activations(
        all_acts_train, 
        important_latent_idx_ranges,
        torch.from_numpy(label_dataset_flat_np),
    )

    rules_by_class = make_rules(important_latent_idx_ranges, important_training_acts, k=topk_rules)

    # %%
    sae_layer_for_rules = f"blocks.{sae_layer_idx}.hook_resid_post"
    num_examples_test = (max_samples_test // batch_size) * batch_size
    idx_test = slice(-num_examples_test, len(clean_labels), 1)
    test_prompts = clean_prompts[idx_test]
    test_labels = clean_labels[idx_test]

    if os.path.exists(acts_test_path):
        all_acts_test = torch.load(acts_test_path, map_location="cpu")
    else:
        all_acts_test = {}
        assert model is not None and saes is not None
        for start_idx in range(0, len(test_prompts), batch_size):
            batch_examples = test_prompts[start_idx : start_idx + batch_size]
            tokens = model.to_tokens(batch_examples)
            all_acts_test[start_idx] = model_run_activations(model, saes, tokens)
        torch.save(all_acts_test, acts_test_path)
        print("testing activations saved to ", acts_test_path)

    # %%
    print(f"---- Testing on {num_examples_test} Examples (batch_size={batch_size}) ----")

    predictions = []
    for start_idx in range(0, len(test_prompts)-batch_size, batch_size):
        activation_batch = all_acts_test[start_idx][sae_layer_for_rules]
        for i in range(batch_size):
            predictions.append(get_prediction(
                activation_batch[i], 
                rules_by_class[sae_layer_for_rules], 
                sae_layer_for_rules, 
                important_latent_idx_ranges,
            ))
            if len(predictions[-1]):
                predictions[-1][0] += 1
                predictions[-1][0] = str(predictions[-1][0])
            
    # for p, gt in zip(predictions, test_labels):
    #     print(len(p), p[0]==gt, type(p[0]), type(gt), len(p) == 1 and p[0] == gt)
    precision = sum(
        1 for p, gt in zip(predictions, test_labels)
        if len(p) == 1 and p[0] == gt
    ) / sum(1 for p in predictions if len(p) == 1)

    abstain = sum(1 for p in predictions if len(p) != 1) / len(test_prompts)

    print(f"Test Precision: {precision*100:.2f}%")
    print(f"Test Abstain: {abstain*100:.2f}%")
    # %%
    rules_as_sets = apply_rule_mask_to_sets_with_ranges(rules_by_class, important_latent_idx_ranges)
    for layer, class_dict in rules_as_sets.items(): 
        print(f"\n=== Layer: {layer} ===")
        for cls, rules in class_dict.items():
            print(f"\nClass: {cls}")
            for idx, rule in enumerate(rules):
                if not rule:
                    print(f"  Rule {idx + 1}: [empty rule]")
                    continue
                explanations = []
                for latent_idx, (lb, ub) in rule:
                    # expl = get_latent_explanation(model_id, expl_layer, latent_idx)
                    explanations.append(f"[{latent_idx}] ∈ ({lb:.3f}, {ub:.3f})")
                readable_rule = " AND ".join(explanations)
                print(f"  Rule {idx + 1}: {readable_rule}")

# %%
if __name__ == '__main__':
    main()