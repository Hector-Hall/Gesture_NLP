from typing import List, Dict, Tuple
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from Gloss_LLM.resized_vocabulary import get_probability_of_a_gloss
ProbabilityDistribution = Dict[str, float]
from Gloss_LLM.constants import MODEL_ID, DTYPE, DEVICE_MAP
from Gloss_LLM.resized_vocabulary import print_prob_table
from Gloss_LLM.resized_vocabulary import compute_next_gloss_probability
from Gloss_LLM.resized_vocabulary import compute_next_gloss_probability_batch
from Gloss_LLM.resized_vocabulary import compute_gloss_probabilities_for_slot
from Gloss_LLM.resized_vocabulary import compute_gloss_probabilities_for_slot_batch

def add_dot_to_sequence(gloss_sequence: List[ProbabilityDistribution],first_dot:int = 1) -> List[ProbabilityDistribution]:
    """
    Adds a dot to each slot of the sequence except the first one.
    The dot will have a probability of 0.0. This is not a problem as we only use the CV probabilities for the first slot.
    """
    new_sequence: List[ProbabilityDistribution] = []
    for i, slot in enumerate(gloss_sequence):
        if i < first_dot:
            new_sequence.append(slot)
        else:
            new_sequence.append({**slot, ".": 0.0})
    return new_sequence


def from_dic_sequence_to_list_sequence(gloss_sequence: List[ProbabilityDistribution],log_proba: bool = False) -> List[List[Tuple[str, float]]]:
    """
    Convert a list of dictionary sequence to a list of list sequence.
    """
    gloss_sequence_log: List[List[Tuple[str, float]]] = []
    for slot in gloss_sequence:
        items = slot.items()
        step = []
        for tok, p in items:
            if log_proba:
                logp = -math.inf if p <= 0.0 else math.log(float(p))
            else:
                logp = -math.inf if p <= 0.0 else p
            step.append((tok, logp))
        gloss_sequence_log.append(step)
    return gloss_sequence_log

def beam_search_slots_given_probability_distribution(gloss_sequence: List[ProbabilityDistribution],beam_size: int = 4,n_best: int = 5) -> List[Tuple[List[str], float, float]]:
    """
    Beam search for len(gloss_sequence) slots.
    - beam_size : amount of sequences to keep
    - n_best : amount of sequences to return
    Return : list of (sequence[str], score_log, proba)
    """

    gloss_sequence_log = from_dic_sequence_to_list_sequence(gloss_sequence,log_proba=True)

    # the beam_size best sequences
    best_sequences: List[Tuple[List[str], float]] = [([], 0.0)]  # (sequence, sum_logp)

    # We look into each slot of the gloss sequence
    for slot in gloss_sequence_log:
        candidates: List[Tuple[List[str], float]] = []
        # We look into each best sequence
        for seq, score in best_sequences:
            # For each best sequence, we add the new token to the sequence and compute the new score
            for tok, logp in slot:
                # As we work with log, we add the log probability (same as multiplying the probabilities)
                candidates.append((seq + [tok], score + logp))
        # Computes the top-k new sequences
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sequences = candidates[:beam_size]

    # Computes the top-n best sequences
    best_sequences.sort(key=lambda x: x[1], reverse=True)

    out = []
    for seq, score in best_sequences[:n_best]:
        # As we work with log, we need to convert the score to a probability
        out.append((seq, score, math.exp(score)))
    return out


def beam_search_slots_with_llm(gloss_sequence: List[ProbabilityDistribution],model: AutoModelForCausalLM,tokenizer: AutoTokenizer, beam_size: int = 4,n_best: int = 5,EOS: int = 1000,device: str = "cpu",length_penalty:float = 1.0) -> List[Tuple[List[str], float, float]]:
    """
    Beam search for len(gloss_sequence) slots.
    - beam_size : amount of sequences to keep
    - n_best : amount of sequences to return
    Return : list of (sequence[str], score_log, proba)
    Uses a length penalty to penalize early stopping.
    """


    # Add dots to each slot of the sequence except the 2 first slots.
    gloss_sequence_with_dots = add_dot_to_sequence(gloss_sequence,first_dot=3)
    gloss_sequence_list = from_dic_sequence_to_list_sequence(gloss_sequence_with_dots,log_proba=False)



    #gloss_sequence_list = from_dic_sequence_to_list_sequence(gloss_sequence,log_proba=False)
    best_sequences: List[Tuple[List[str], float,float]] = [([], 0.0,0.0)]  # (sequence, sum_logp,rerank)
    slot_index = 0
    for slot in gloss_sequence_list:
        candidates: List[Tuple[List[str], float,float]] = []
        # The length penalty score for this slot
        length_penalty_score = ((5+(slot_index+1))**length_penalty) / (6**length_penalty)
        
        # We look into each best sequence
        for seq, score,_ in best_sequences:

            # if the index is 0 we use the probability distribution from the CV model
            if slot_index == 0:
                for token, proba in slot:
                    if proba == 0.0:
                        candidates.append((seq + [token], -math.inf, -math.inf))
                    else:
                        new_score = score + math.log(proba)
                        # The rerank is used to rank the sequences with the length penalty
                        rerank = new_score / length_penalty_score
                        candidates.append((seq + [token], new_score,rerank))
            else:
                # if the last token is a dot, keep the sequence but don't extend it
                if seq[-1] == ".":
                    candidates.append((seq, score, score / length_penalty_score))
                    continue
                # if the index is not 0 we use the probability distribution from the LLM model
                # We only use the glosses in the spot. We don't need the probabilities of the whole vocabulary.
                possible_glosses_in_slot = [token for token, _ in slot]
                
                # Compute probabilities for all glosses (handles multi-token properly)
                gloss_probs = compute_gloss_probabilities_for_slot(
                    prefix_sequence=seq,
                    candidate_glosses=possible_glosses_in_slot,
                    tok=tokenizer,
                    model=model,
                    EOS=EOS,
                    device=device
                )
                
                # we compute the probability of each token in the slot
                for token, proba in slot:
                    llm_proba = gloss_probs.get(token, 0.0)
                    if llm_proba == 0.0:
                        candidates.append((seq + [token], -math.inf, -math.inf))
                    else:
                        new_score = score + math.log(llm_proba)
                        # The rerank is used to rank the sequences with the length penalty
                        rerank = new_score / length_penalty_score
                        candidates.append((seq + [token], new_score,rerank))
            
        # Computes the top-k new sequences based on the rerank score
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_sequences = candidates[:beam_size]
        if slot_index == 3:
            print(best_sequences)
        slot_index += 1
    
    # Computes the top-n best sequences based on the rerank score
    best_sequences.sort(key=lambda x: x[2], reverse=True)

    out = []
    for seq, score,_ in best_sequences[:n_best]:
        # As we work with log, we need to convert the score to a probability
        out.append((seq, score, math.exp(score)))
    return out


def beam_search_slots_with_llm_batched(gloss_sequence: List[ProbabilityDistribution], model: AutoModelForCausalLM,
                                       tokenizer: AutoTokenizer, beam_size: int = 4, n_best: int = 5,
                                       EOS: int = 1000, device: str = "cpu", length_penalty: float = 1.0) -> List[Tuple[List[str], float, float]]:
    """
    OPTIMIZED: Batched beam search for len(gloss_sequence) slots.
    Processes all beam candidates in a single forward pass for ~4-5x speedup.
    
    - beam_size : amount of sequences to keep
    - n_best : amount of sequences to return
    Return : list of (sequence[str], score_log, proba)
    Uses a length penalty to penalize early stopping.
    """
    import torch
    
    # Add dots to each slot of the sequence except the 2 first slots.
    gloss_sequence_with_dots = add_dot_to_sequence(gloss_sequence, first_dot=3)
    gloss_sequence_list = from_dic_sequence_to_list_sequence(gloss_sequence_with_dots, log_proba=False)

    best_sequences: List[Tuple[List[str], float, float]] = [([], 0.0, 0.0)]  # (sequence, sum_logp, rerank)
    slot_index = 0
    
    # Import allowed_token_ids function for caching
    from Gloss_LLM.resized_vocabulary import allowed_token_ids as compute_allowed_token_ids
    
    for slot in gloss_sequence_list:
        candidates: List[Tuple[List[str], float, float]] = []
        # The length penalty score for this slot
        length_penalty_score = ((5 + (slot_index + 1)) ** length_penalty) / (6 ** length_penalty)
        
        # if the index is 0 we use the probability distribution from the CV model
        if slot_index == 0:
            for seq, score, _ in best_sequences:
                for token, proba in slot:
                    if proba == 0.0:
                        candidates.append((seq + [token], -math.inf, -math.inf))
                    else:
                        new_score = score + math.log(proba)
                        rerank = new_score / length_penalty_score
                        candidates.append((seq + [token], new_score, rerank))
        else:
            # Filter out sequences that ended with a dot
            active_sequences = [(seq, score, rerank) for seq, score, rerank in best_sequences if seq[-1] != "."]
            
            if not active_sequences:
                # All sequences have terminated
                candidates = best_sequences
            else:
                # BATCH PROCESSING: Compute probabilities for all active sequences at once
                possible_glosses_in_slot = [token for token, _ in slot]
                active_seqs_only = [seq for seq, _, _ in active_sequences]
                
                # Pre-compute allowed token IDs for this slot (expensive, so compute once)
                cached_allowed_ids, _ = compute_allowed_token_ids(possible_glosses_in_slot, tokenizer, EOS)
                
                # Get batched probabilities for all sequences (handles multi-token glosses)
                batch_gloss_probs = compute_gloss_probabilities_for_slot_batch(
                    active_seqs_only, possible_glosses_in_slot, tokenizer, model, EOS, device,
                    cached_allowed_ids=cached_allowed_ids
                )
                
                # Now expand each sequence with its corresponding probabilities
                for idx, (seq, score, _) in enumerate(active_sequences):
                    gloss_probs = batch_gloss_probs[idx]  # Get probabilities for this sequence
                    
                    for token, proba in slot:
                        llm_proba = gloss_probs.get(token, 0.0)
                        if llm_proba == 0.0:
                            candidates.append((seq + [token], -math.inf, -math.inf))
                        else:
                            new_score = score + math.log(llm_proba)
                            rerank = new_score / length_penalty_score
                            candidates.append((seq + [token], new_score, rerank))
                
                # Add terminated sequences back
                terminated = [(seq, score, rerank) for seq, score, rerank in best_sequences if seq[-1] == "."]
                candidates.extend(terminated)
        
        # Computes the top-k new sequences based on the rerank score
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_sequences = candidates[:beam_size]
        
        if slot_index == 3:
            print(best_sequences)
        slot_index += 1
    
    # Computes the top-n best sequences based on the rerank score
    best_sequences.sort(key=lambda x: x[2], reverse=True)

    out = []
    for seq, score, _ in best_sequences[:n_best]:
        # As we work with log, we need to convert the score to a probability
        out.append((seq, score, math.exp(score)))
    return out


def beam_search_slots_with_llm_multitoken(gloss_sequence: List[ProbabilityDistribution], model: AutoModelForCausalLM,
                                          tokenizer: AutoTokenizer, beam_size: int = 4, n_best: int = 5,
                                          EOS: int = 1000, device: str = "cpu", length_penalty: float = 1.0) -> List[Tuple[List[str], float, float]]:
    """
    IMPROVED: Beam search that properly handles multi-token glosses.
    
    This version computes proper sequence probabilities for multi-token glosses
    (e.g., "thank you", "sign language") instead of returning 0.0.
    
    - beam_size : amount of sequences to keep
    - n_best : amount of sequences to return
    Return : list of (sequence[str], score_log, proba)
    Uses a length penalty to penalize early stopping.
    """
    
    # Add dots to each slot of the sequence except the 2 first slots.
    gloss_sequence_with_dots = add_dot_to_sequence(gloss_sequence, first_dot=3)
    gloss_sequence_list = from_dic_sequence_to_list_sequence(gloss_sequence_with_dots, log_proba=False)

    best_sequences: List[Tuple[List[str], float, float]] = [([], 0.0, 0.0)]  # (sequence, sum_logp, rerank)
    slot_index = 0
    
    for slot in gloss_sequence_list:
        candidates: List[Tuple[List[str], float, float]] = []
        # The length penalty score for this slot
        length_penalty_score = ((5 + (slot_index + 1)) ** length_penalty) / (6 ** length_penalty)
        
        # if the index is 0 we use the probability distribution from the CV model
        if slot_index == 0:
            for seq, score, _ in best_sequences:
                for token, proba in slot:
                    if proba == 0.0:
                        candidates.append((seq + [token], -math.inf, -math.inf))
                    else:
                        new_score = score + math.log(proba)
                        rerank = new_score / length_penalty_score
                        candidates.append((seq + [token], new_score, rerank))
        else:
            # Process each sequence in the beam
            for seq, score, _ in best_sequences:
                # if the last token is a dot, we don't add a new token to the sequence
                if seq[-1] == ".":
                    continue
                
                # Get candidate glosses for this slot
                possible_glosses_in_slot = [token for token, _ in slot]
                
                # Compute probabilities for all glosses (handles multi-token properly)
                gloss_probs = compute_gloss_probabilities_for_slot(
                    prefix_sequence=seq,
                    candidate_glosses=possible_glosses_in_slot,
                    tok=tokenizer,
                    model=model,
                    EOS=EOS,
                    device=device
                )
                
                # Expand candidates with computed probabilities
                for token, cv_proba in slot:
                    llm_proba = gloss_probs.get(token, 0.0)
                    
                    if llm_proba == 0.0:
                        candidates.append((seq + [token], -math.inf, -math.inf))
                    else:
                        new_score = score + math.log(llm_proba)
                        rerank = new_score / length_penalty_score
                        candidates.append((seq + [token], new_score, rerank))
        
        # Computes the top-k new sequences based on the rerank score
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_sequences = candidates[:beam_size]
        
        if slot_index == 3:
            print(best_sequences)
        slot_index += 1
    
    # Computes the top-n best sequences based on the rerank score
    best_sequences.sort(key=lambda x: x[2], reverse=True)

    out = []
    for seq, score, _ in best_sequences[:n_best]:
        # As we work with log, we need to convert the score to a probability
        out.append((seq, score, math.exp(score)))
    return out


# Example
if __name__ == "__main__":
    slots = [
        {"i": 0.4, "you": 0.25, "we": 0.15, "they": 0.1, "he": 0.06, "she": 0.04},
        {"want": 0.5, "like": 0.2, "need": 0.15, "eat": 0.1, "go": 0.03, "watch": 0.02},
        {"a": 0.45, "the": 0.25, "some": 0.15, "to": 0.1, "of": 0.03, "my": 0.02},
        {"pizza": 0.35, "movie": 0.25, "restaurant": 0.2, "water": 0.1, "ring": 0.06, "airplane": 0.04},
        {"now": 0.3, "today": 0.25, "please": 0.2, "soon": 0.15, "outside": 0.06, "house": 0.04},
    ]

    """tops = beam_search_slots_given_probability_distribution(slots, beam_size=4, n_best=3)
    for i, (seq, s_log, p) in enumerate(tops, 1):
        print(f"{i:>2}. {' '.join(seq):50s}  logP={s_log:.4f}  P={p:.6f}")"""
    
    
    
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE_MAP)
    model.eval()
    EOS = tok.eos_token_id

    device = model.device

    # Benchmark both versions
    import time
    
    print("=" * 80)
    print("ORIGINAL (Sequential) Beam Search:")
    print("=" * 80)
    start_time = time.time()
    tops_original = beam_search_slots_with_llm(slots, model, tok, beam_size=4, n_best=3, EOS=EOS, device=device)
    original_time = time.time() - start_time
    
    for i, (seq, s_log, p) in enumerate(tops_original, 1):
        print(f"{i:>2}. {' '.join(seq):50s}  logP={s_log:.4f}  P={p:.6f}")
    print(f"\nTime: {original_time:.3f}s")
    
    print("\n" + "=" * 80)
    print("OPTIMIZED (Batched) Beam Search:")
    print("=" * 80)
    start_time = time.time()
    tops_batched = beam_search_slots_with_llm_batched(slots, model, tok, beam_size=4, n_best=3, EOS=EOS, device=device)
    batched_time = time.time() - start_time
    
    for i, (seq, s_log, p) in enumerate(tops_batched, 1):
        print(f"{i:>2}. {' '.join(seq):50s}  logP={s_log:.4f}  P={p:.6f}")
    print(f"\nTime: {batched_time:.3f}s")
    
    print("\n" + "=" * 80)
    print(f"Speedup: {original_time / batched_time:.2f}x faster")
    print("=" * 80)