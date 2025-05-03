'''
Loading, running, and evaluating models/SAEs, including API explanations
'''

import torch
from sae_lens import SAE, HookedSAETransformer
import requests
from typing import Tuple
from tqdm import tqdm

def load_model_and_saes(model_name: str, sae_layers: list, device: str, hf_cache: str):
    print("------------ Loading Model ------------")
    print(f"Loading model: {model_name}...")
    model = HookedSAETransformer.from_pretrained(model_name, device=device, cache_dir=hf_cache)
    for param in model.parameters():
        param.requires_grad_(False)
    print("Model loaded and parameters frozen.")
    saes = [
        SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=device
        )[0] for layer in sae_layers
    ]
    return model, saes

def load_igmask(file_path):
    """Load an IGMask instance from a file."""
    state_dict = torch.load(file_path)

    # Ensure the loaded state contains the necessary components
    if 'ig_scores' not in state_dict:
        raise ValueError("The saved file does not contain the necessary components for IGMask.")

    # Recreate the IGMask instance
    igmask = IGMask(ig_scores=state_dict['ig_scores'])
    return igmask

def build_sae_hook_fn(
    # Core components
    sae,
    sequence,
    bos_token_id,

    # Masking options
    circuit_mask=None,
    use_mask=False,
    binarize_mask=False,
    mean_mask=False,
    ig_mask_threshold=None,

    # Caching behavior
    cache_sae_grads=False,
    cache_masked_activations=False,
    cache_sae_activations=False,

    # Ablation options
    mean_ablate=False,  # Controls mean ablation of the SAE
    fake_activations=False,  # Controls whether to use fake activations
):    # make the mask for the sequence
    mask = torch.ones_like(sequence, dtype=torch.bool)
    # mask[sequence == pad_token_id] = False
    mask[sequence == bos_token_id] = False # where mask is false, keep original
    def sae_hook(value, hook):
        # print(f"sae {sae.cfg.hook_name} running at layer {hook.layer()}")
        feature_acts = sae.encode(value)
        feature_acts = feature_acts * mask.unsqueeze(-1)
        if fake_activations != False and sae.cfg.hook_layer == fake_activations[0]:
            feature_acts = fake_activations[1]
        if cache_sae_grads:
            raise NotImplementedError("torch is confusing")
            sae.feature_acts = feature_acts.requires_grad_(True)
            sae.feature_acts.retain_grad()

        if cache_sae_activations:
            sae.feature_acts = feature_acts.detach().clone()

        # Learned Binary Masking
        if use_mask:
            if mean_mask:
                # apply the mask, with mean ablations
                feature_acts = sae.mask(feature_acts, binary=binarize_mask, mean_ablation=sae.mean_ablation)
            else:
                # apply the mask, without mean ablations
                feature_acts = sae.mask(feature_acts, binary=binarize_mask)

        # IG Masking
        if ig_mask_threshold != None:
            # apply the ig mask
            if mean_mask:
                feature_acts = sae.igmask(feature_acts, threshold=ig_mask_threshold, mean_ablation=sae.mean_ablation)
            else:
                feature_acts = sae.igmask(feature_acts, threshold=ig_mask_threshold)

        if circuit_mask is not None:
            hook_point = sae.cfg.hook_name
            if mean_mask==True:
                feature_acts = circuit_mask(feature_acts, hook_point, mean_ablation=sae.mean_ablation)
            else:
                feature_acts = circuit_mask(feature_acts, hook_point)

        if cache_masked_activations:
            sae.feature_acts = feature_acts.detach().clone()
        if mean_ablate:
            feature_acts = sae.mean_ablation

        out = sae.decode(feature_acts)
        # choose out or value based on the mask
        mask_expanded = mask.unsqueeze(-1).expand_as(value)
        value = torch.where(mask_expanded, out, value)
        return value
    return sae_hook

def build_hooks_list(sequence,
                    saes,
                    bos_token_id,
                    cache_sae_activations=False,
                    cache_sae_grads=False,
                    circuit_mask=None,
                    use_mask=False,
                    binarize_mask=False,
                    mean_mask=False,
                    cache_masked_activations=False,
                    mean_ablate=False,
                    fake_activations: Tuple[int, torch.Tensor] = False,
                    ig_mask_threshold=None,
):
    hooks = []
    for sae in saes:
        hooks.append(
            (
                sae.cfg.hook_name,
                build_sae_hook_fn(sae, sequence, bos_token_id, cache_sae_grads=cache_sae_grads, circuit_mask=circuit_mask, use_mask=use_mask, binarize_mask=binarize_mask, cache_masked_activations=cache_masked_activations, cache_sae_activations=cache_sae_activations, mean_mask=mean_mask, mean_ablate=mean_ablate, fake_activations=fake_activations, ig_mask_threshold=ig_mask_threshold),
            )
        )
    return hooks

def model_sae_forward(model, saes, batch):
    bos_token_id = model.tokenizer.bos_token_id
    return model.run_with_hooks(
        batch,
        return_type="logits",
        fwd_hooks=build_hooks_list(
            batch,
            saes,
            bos_token_id,
            use_mask=False,
            mean_mask=True,
            binarize_mask=True,
            cache_masked_activations=False,
        )
    )

def model_run_activations(model, saes, batch):
    bos_token_id = model.tokenizer.bos_token_id
    model.run_with_hooks(
        batch,
        return_type="logits",
        fwd_hooks=build_hooks_list(
            batch,
            saes,
            bos_token_id,
            use_mask=False,
            mean_mask=False,
            binarize_mask=True,
            cache_masked_activations=True
        )
    )
    batch_acts = {}
    for sae in saes:
        act = sae.feature_acts.detach().cpu()
        batch_acts[sae.cfg.hook_name] = act[:,-1,:]
        del sae.feature_acts
    model.reset_hooks(including_permanent=True)
    return batch_acts

def logit_diff_fn(logits, clean_labels, corr_labels, token_wise: bool = False):
    clean_logits = logits[torch.arange(logits.shape[0]), -1, clean_labels]
    corr_logits = logits[torch.arange(logits.shape[0]), -1, corr_labels]
    return (clean_logits - corr_logits).mean() if not token_wise else (clean_logits - corr_logits)


def evaluate_on_task(model, saes, prompt_dataset, label_dataset):
    all_preds = []
    with torch.no_grad():
        for i in tqdm(range(prompt_dataset.shape[0])):
            prompts = prompt_dataset[i]
            outputs = model_sae_forward(model, saes, prompts)
            preds = outputs[:,-1].argmax(dim=-1)
            all_preds.append(preds)
            del outputs
            torch.cuda.empty_cache()
    all_preds = torch.stack(all_preds)
        
    for pred_ids, label_ids in zip(all_preds, label_dataset):
        for x in pred_ids:
            print(model.tokenizer.decode(x, skip_special_tokens=True))
        print("---")
        for x in label_ids:
            print(model.tokenizer.decode(x, skip_special_tokens=True))

    accuracy = (all_preds.cpu() == label_dataset.cpu()).float().mean().item()
    return accuracy

def get_latent_explanation(model_id: str, layer: str, index: int) -> str:
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}/{index}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f'{index}: {data["explanations"][0]["description"]}'
    else:
        print(f"Error: {response.status_code}")
        return None


def get_rule_explanation(model_id: str, layer: str, rule_set: list) -> str:
    explanation_text = ""
    for latent in rule_set:
        explanation = get_latent_explanation(model_id, layer, latent)
        if explanation:
            explanation_text += explanation + '\n'
    return explanation_text
