import torch
from torch import Tensor
from jaxtyping import Float
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM
from Gloss_LLM.constants import MODEL_ID, GLOSSES
from typing import Set
from Gloss_LLM.constants import DTYPE, DEVICE_MAP
from Gloss_LLM.gloss_encoding import encode_gloss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions as CausalLMOutput
from transformers import PreTrainedModel
import math

def allowed_sequences(glosses: list[str],tok: PreTrainedTokenizerBase)->Set[tuple[int, ...]]:
    """
    Compute all the allowed sequences for the glosses.
    For a gloss we allow "gloss" and " gloss".
    """
    allowed_sequences: Set[tuple[int, ...]] = set()
    for w in glosses:
        ids_first = tuple(encode_gloss(w,tok))
        ids_after = tuple(encode_gloss(" " + w,tok))
        allowed_sequences.add(ids_first)
        allowed_sequences.add(ids_after)
    return allowed_sequences

def allowed_token_ids(glosses: list[str],tok: PreTrainedTokenizerBase,EOS: int)->tuple[list[int],Set[tuple[int, ...]]]:
    """
    Compute the allowed token IDs for the glosses.
    """
    sequences = allowed_sequences(glosses,tok)
    gloss_ids = sorted({tid for seq in sequences for tid in seq})
    return sorted(set(gloss_ids) | {EOS}), sequences


def token_str(tid: int,EOS: int,tok: PreTrainedTokenizerBase) -> str:
    """
    Convert a token ID to a string.
    """
    if tid == EOS:
        return "<eos>"
    # convert_ids_to_tokens montre le préfixe d'espace explicitement; decode([tid]) donne le rendu textuel
    s :str= tok.convert_ids_to_tokens(tid)
    # Affiche aussi la variante décodée (utile pour vérifier les espaces)
    try:
        d = tok.decode([tid])
        if d != s:
            s = f"{s} | {repr(d)}"
    except Exception:
        pass
    return s

def print_prob_table(ids:list[int], probs, EOS: int,tok: PreTrainedTokenizerBase, top_k:int|None=None)->list[tuple[int, float]]:
    """
    Print the probability table for the next gloss.
    """
    # ids: list of allowed IDs, probs: tensor [vocab] already softmaxed after masking
    rows :list[tuple[int, float]] = []
    for tid in ids:
        rows.append((tid, probs[tid].item()))
    rows.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        rows = rows[:top_k]
    width = max(len(token_str(t,EOS,tok)) for t, _ in rows) if rows else 10
    for tid, p in rows:
        print(f"{token_str(tid,EOS,tok):<{width}}  {p: .6f}")
    return rows


def compute_next_gloss_probability(gloss_sequence: list[str],glosses: list[str],tok: PreTrainedTokenizerBase,\
                                   model: PreTrainedModel,EOS: int,device: torch.device=torch.device("cpu"))->\
                                    tuple:
    """
    Compute the probabilities for the next gloss given a gloss_sequence.
    """
    # Join the sequence into a single string for proper tokenization
    gloss_string = " ".join(gloss_sequence) if isinstance(gloss_sequence, list) else gloss_sequence
    
    inputs :dict = tok(gloss_string, return_tensors="pt").to(device)
    # computes the logits for the next token
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[:, -1, :]
        logits = logits.float()

    # masks the logits for the non-allowed tokens
    allow_token_ids, allowed_sequences = allowed_token_ids(glosses,tok,EOS)
    mask = torch.full_like(logits, float("-inf"))
    mask[0, allow_token_ids] = 0.0
    masked_logits = logits + mask

    # numerical stability
    masked_logits = torch.nan_to_num(masked_logits, nan=-1e9, posinf=1e9, neginf=-1e9)

    # stable softmax (subtract max)
    masked_logits = masked_logits - masked_logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(masked_logits, dim=-1)
    return probs, allow_token_ids, inputs


def compute_next_gloss_probability_batch(gloss_sequences: list[list[str]], glosses: list[str], 
                                        tok: PreTrainedTokenizerBase, model: PreTrainedModel, 
                                        EOS: int, device: torch.device = torch.device("cpu"),
                                        cached_allowed_ids: tuple = None) -> \
                                        tuple:
    """
    Compute the probabilities for the next gloss given a batch of gloss_sequences.
    More efficient than calling compute_next_gloss_probability in a loop.
    
    Args:
        gloss_sequences: List of gloss sequences (each is a list of strings)
        glosses: List of allowed glosses
        tok: Tokenizer
        model: LLM model
        EOS: End of sequence token ID
        device: Device to run on
        cached_allowed_ids: Pre-computed (allow_token_ids, allowed_sequences) tuple for efficiency
        
    Returns:
        probs: Tensor of shape (batch_size, vocab_size) with probabilities
        allow_token_ids: List of allowed token IDs
    """
    if not gloss_sequences:
        raise ValueError("gloss_sequences cannot be empty")
    
    # Convert list of token lists to list of strings for tokenization
    gloss_strings = [" ".join(seq) for seq in gloss_sequences]
    
    # Tokenize all sequences at once with padding (left padding for causal models)
    inputs = tok(gloss_strings, return_tensors="pt", padding=True).to(device)
    
    # Set pad_token_id if not set
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    
    # Compute logits for all sequences in one forward pass
    with torch.no_grad():
        out = model(**inputs)
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # -1 for 0-indexing
        batch_indices = torch.arange(len(gloss_sequences), device=device)
        logits = out.logits[batch_indices, seq_lengths, :]  # (batch_size, vocab_size)
        logits = logits.float()

    # Mask the logits for non-allowed tokens (use cached if available)
    if cached_allowed_ids is not None:
        allow_token_ids = cached_allowed_ids
    else:
        allow_token_ids, _ = allowed_token_ids(glosses, tok, EOS)
    
    mask = torch.full((1, logits.size(-1)), float("-inf"), device=device)
    mask[0, allow_token_ids] = 0.0
    masked_logits = logits + mask

    # Numerical stability
    masked_logits = torch.nan_to_num(masked_logits, nan=-1e9, posinf=1e9, neginf=-1e9)

    # Stable softmax (subtract max per sequence)
    masked_logits = masked_logits - masked_logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(masked_logits, dim=-1)
    
    return probs, allow_token_ids


   
def get_probability_of_a_gloss(gloss: str,tok: PreTrainedTokenizerBase,probs)->float:
    """
    Get the probability of a gloss (IMPORTANT: only works for single-token glosses).
    Returns 0.0 for multi-token glosses.
    Use get_probability_of_a_gloss_multitoken() for proper multi-token support.
    """
    gloss_ids = encode_gloss(gloss,tok)
    if len(gloss_ids) != 1:
        return 0.0
    else:
        # Handle both 1D (batch_probs[idx]) and 2D (probs) tensors
        if probs.dim() == 1:
            return probs[gloss_ids].item()
        else:
            return probs[0, gloss_ids].item()


def get_probability_of_a_gloss_multitoken(gloss: str, prefix_sequence: list[str], 
                                          tok: PreTrainedTokenizerBase, model: PreTrainedModel,
                                          EOS: int, device: torch.device) -> float:
    """
    Get the probability of a gloss, properly handling multi-token glosses.
    
    For single-token glosses: Returns P(gloss | prefix)
    For multi-token glosses: Returns P(token1) * P(token2 | token1) * ... (autoregressive)
    
    Args:
        gloss: The gloss to compute probability for (e.g., "thank you")
        prefix_sequence: List of glosses that come before (e.g., ["i", "want"])
        tok: Tokenizer
        model: LLM model
        EOS: End of sequence token ID
        device: Device to run on
        
    Returns:
        Probability of the gloss given the prefix (product of token probabilities)
    """
    gloss_ids = encode_gloss(gloss, tok)
    
    if len(gloss_ids) == 1:
        # Single token case - join prefix sequence into string for proper tokenization
        prefix_string = " ".join(prefix_sequence) if prefix_sequence else ""
        inputs = tok(prefix_string, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            return probs[0, gloss_ids[0]].item()
    
    # Multi-token case - compute sequence probability autoregressively
    total_log_prob = 0.0
    current_sequence = prefix_sequence.copy()
    
    # We need to add each token one by one and compute its probability
    # Note: This requires predicting the gloss as a sequence
    # For now, we'll tokenize the full sequence and compute conditional probs
    
    # Get token representations with and without space prefix
    gloss_with_space = " " + gloss
    ids_no_space = tok.encode(gloss, add_special_tokens=False)
    ids_with_space = tok.encode(gloss_with_space, add_special_tokens=False)
    
    # Use the appropriate tokenization based on context
    # After existing tokens, we typically need the space-prefixed version
    token_ids = ids_with_space if len(prefix_sequence) > 0 else ids_no_space
    
    # Compute probability for each token in sequence
    for i, token_id in enumerate(token_ids):
        # Tokenize current sequence - join into string for proper tokenization
        current_string = " ".join(current_sequence) if current_sequence else ""
        inputs = tok(current_string, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            token_prob = probs[0, token_id].item()
            
            if token_prob <= 0.0:
                return 0.0  # If any token has zero probability, the whole sequence does
            
            total_log_prob += math.log(token_prob)
        
        # Add the predicted token to the sequence for next iteration
        # Decode the token to text to maintain consistency
        token_text = tok.decode([token_id])
        current_sequence.append(token_text)
    
    # Return the product of probabilities (exp of sum of log probs)
    return math.exp(total_log_prob)


def compute_gloss_probabilities_for_slot(prefix_sequence: list[str], candidate_glosses: list[str],
                                        tok: PreTrainedTokenizerBase, model: PreTrainedModel,
                                        EOS: int, device: torch.device) -> dict[str, float]:
    """
    Compute probabilities for all candidate glosses in a slot, handling multi-token glosses properly.
    
    This is a more efficient batch implementation that groups single-token and multi-token glosses.
    
    Args:
        prefix_sequence: List of glosses that come before
        candidate_glosses: List of possible glosses for this slot
        tok: Tokenizer
        model: LLM model
        EOS: End of sequence token ID
        device: Device to run on
        
    Returns:
        Dictionary mapping gloss -> probability
    """
    result = {}
    
    # Separate single-token and multi-token glosses
    single_token_glosses = []
    multi_token_glosses = []
    
    for gloss in candidate_glosses:
        gloss_ids = encode_gloss(gloss, tok)
        if len(gloss_ids) == 1:
            single_token_glosses.append(gloss)
        else:
            multi_token_glosses.append(gloss)
    
    # Batch process single-token glosses (efficient) with masking for consistency
    if single_token_glosses:
        # Join prefix sequence into string for proper tokenization
        prefix_string = " ".join(prefix_sequence) if prefix_sequence else ""
        inputs = tok(prefix_string, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :].float()
            
            # Mask logits to only allowed glosses (same as batched version)
            allow_token_ids, _ = allowed_token_ids(candidate_glosses, tok, EOS)
            mask = torch.full_like(logits, float("-inf"))
            mask[0, allow_token_ids] = 0.0
            masked_logits = logits + mask
            
            # Numerical stability
            masked_logits = torch.nan_to_num(masked_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            masked_logits = masked_logits - masked_logits.max(dim=-1, keepdim=True).values
            
            probs = torch.softmax(masked_logits, dim=-1)
            
            for gloss in single_token_glosses:
                gloss_id = encode_gloss(gloss, tok)[0]
                result[gloss] = probs[0, gloss_id].item()
    
    # Process multi-token glosses individually (necessary for sequence probability)
    for gloss in multi_token_glosses:
        result[gloss] = get_probability_of_a_gloss_multitoken(
            gloss, prefix_sequence, tok, model, EOS, device
        )
    
    return result


def compute_gloss_probabilities_for_slot_batch(prefix_sequences: list[list[str]], 
                                              candidate_glosses: list[str],
                                              tok: PreTrainedTokenizerBase, 
                                              model: PreTrainedModel,
                                              EOS: int, 
                                              device: torch.device,
                                              cached_allowed_ids: tuple = None) -> list[dict[str, float]]:
    """
    Batched version of compute_gloss_probabilities_for_slot.
    Computes probabilities for all candidate glosses for multiple prefix sequences at once.
    Properly handles multi-token glosses.
    
    Args:
        prefix_sequences: List of prefix sequences (each is a list of glosses)
        candidate_glosses: List of possible glosses for this slot
        tok: Tokenizer
        model: LLM model
        EOS: End of sequence token ID
        device: Device to run on
        cached_allowed_ids: Pre-computed allowed token IDs for efficiency
        
    Returns:
        List of dictionaries (one per prefix), each mapping gloss -> probability
    """
    import math
    
    # Separate single-token and multi-token glosses
    single_token_glosses = []
    multi_token_glosses = []
    
    for gloss in candidate_glosses:
        gloss_ids = encode_gloss(gloss, tok)
        if len(gloss_ids) == 1:
            single_token_glosses.append(gloss)
        else:
            multi_token_glosses.append(gloss)
    
    # Initialize results for each sequence
    results = [{} for _ in prefix_sequences]
    
    # Batch process single-token glosses (efficient)
    if single_token_glosses:
        # Get batched probabilities using existing function
        batch_probs, _ = compute_next_gloss_probability_batch(
            prefix_sequences, candidate_glosses, tok, model, EOS, device, cached_allowed_ids
        )
        
        # Extract probabilities for each sequence
        for idx, probs in enumerate(batch_probs):
            for gloss in single_token_glosses:
                gloss_id = encode_gloss(gloss, tok)[0]
                results[idx][gloss] = probs[gloss_id].item()
    
    # Process multi-token glosses (need to do individually for each sequence)
    if multi_token_glosses:
        for idx, prefix_seq in enumerate(prefix_sequences):
            for gloss in multi_token_glosses:
                results[idx][gloss] = get_probability_of_a_gloss_multitoken(
                    gloss, prefix_seq, tok, model, EOS, device
                )
    
    return results


def prompt_testing(prompt:list[str])-> None:
    tok = PreTrainedTokenizerBase.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE_MAP)
    model.eval()
    EOS = tok.eos_token_id

    device = model.device
    probs, allowed_token_ids, inputs = compute_next_gloss_probability(prompt,GLOSSES,tok,model,EOS,device)



    print("=== Next-token probabilities (over GLOSSES set) ===")
    rows = print_prob_table(allowed_token_ids, probs[0], EOS, tok, top_k=None)  # set top_k=20 if you want to truncate

    # ------------ Constrained sampling (optional) ------------
    # We sample only from the allowed IDs (distribution already renormalized)
    allowed_tensor = torch.tensor(allowed_token_ids, device=probs.device)
    allowed_probs = probs[0, allowed_tensor]
    next_index_in_allowed = torch.multinomial(allowed_probs, num_samples=1)
    next_tid = allowed_tensor[next_index_in_allowed].item()
    next_prob = allowed_probs[next_index_in_allowed].item()

    print("\nSampled next token:")
    print(f"  id={next_tid}, tok={token_str(int(next_tid),EOS,tok)}, p={next_prob:.6f}")

    # ------------ Concatenate and display the extended sequence ------------
    new_input_ids = torch.cat([inputs["input_ids"], torch.tensor([[next_tid]], device=inputs["input_ids"].device)], dim=-1)
    print("\nDecoded so far:")
    print(tok.decode(new_input_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    prompt_testing(prompt=["me", "want", "eat"])