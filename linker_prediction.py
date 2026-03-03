import argparse
import sys
import random
import torch
import torch.nn.functional as F
from transformers import EsmTokenizer, EsmForMaskedLM
from tqdm import tqdm

# Standard amino acids for initialization
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def get_sequence_pll(full_sequence, linker_start_idx, linker_length, model, tokenizer):
    """
    Calculates the Pseudo-Log-Likelihood (PLL) of the linker region.
    It rigorously masks each position in the linker one by one and calculates 
    the log probability of the actual amino acid chosen, giving an overall sequence score.
    """
    pll = 0.0
    sequence_list = list(full_sequence)
    
    for i in range(linker_length):
        target_idx = linker_start_idx + i
        original_aa = sequence_list[target_idx]
        
        # Mask the single position
        sequence_list[target_idx] = tokenizer.mask_token
        masked_seq = "".join(sequence_list)
        
        inputs = tokenizer(masked_seq, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            
        # Find mask token index in the tensor
        mask_tensor_idx = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1][0]
        
        # Get log probability of the generated amino acid
        original_token_id = tokenizer.convert_tokens_to_ids(original_aa)
        log_probs = F.log_softmax(logits[0, mask_tensor_idx, :], dim=-1)
        pll += log_probs[original_token_id].item()
        
        # Restore the amino acid for the next step
        sequence_list[target_idx] = original_aa
        
    return pll

def design_linker_gibbs(domain_a, domain_b, linker_length, num_seqs=10, 
                        masks_per_step=2, steps=50, temperature=1.0, 
                        model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Generates multiple linker candidates using Gibbs Sampling and ranks them by overall PLL.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()

    candidates = []
    linker_start_idx = len(domain_a)

    print(f"\nGenerating {num_seqs} candidates using Gibbs Sampling...")
    print(f"(Masking {masks_per_step} residues per step for {steps} steps per sequence)")
    
    for seq_idx in tqdm(range(num_seqs), desc="Generating Candidates"):
        # 1. Initialize with random amino acids (Burn-in phase start)
        current_linker = [random.choice(AMINO_ACIDS) for _ in range(linker_length)]
        
        # 2. Gibbs sampling Markov chain
        for step in range(steps):
            # Randomly pick number of masks
            mask_count = random.randint(1, masks_per_step)
            # Randomly pick positions to mask
            mask_indices = random.sample(range(linker_length), min(mask_count, linker_length))
            
            # Apply masks
            for idx in mask_indices:
                current_linker[idx] = tokenizer.mask_token
            full_sequence = domain_a + "".join(current_linker) + domain_b
            inputs = tokenizer(full_sequence, return_tensors="pt")
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            mask_tensor_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
            
            # 3. Sample new amino acids based on predictions
            for i, tensor_idx in enumerate(mask_tensor_indices):
                # Apply temperature to flatten or sharpen the distribution
                scaled_logits = logits[0, tensor_idx, :] / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                
                # Sample from the probability distribution
                sampled_token_id = torch.multinomial(probs, 1).item()
                sampled_aa = tokenizer.decode([sampled_token_id]).strip()
                
                # Fallback to random AA if the model predicts special tokens (like <cls>)
                if sampled_aa not in AMINO_ACIDS:
                    sampled_aa = random.choice(AMINO_ACIDS)
                    
                current_linker[mask_indices[i]] = sampled_aa

        # 4. Finalize candidate and evaluate overall probability
        final_linker = "".join(current_linker)
        final_protein = domain_a + final_linker + domain_b
        
        pll_score = get_sequence_pll(final_protein, linker_start_idx, linker_length, model, tokenizer)
        
        candidates.append({
            "linker": final_linker,
            "protein": final_protein,
            "pll": pll_score
        })
        print(f"Candidate {seq_idx+1}/{num_seqs}: {final_linker} (Score: {pll_score:.2f})")


    # 5. Rank the candidates from most probable to least probable
    # (Higher PLL, i.e., closer to 0, is better)
    candidates.sort(key=lambda x: x["pll"], reverse=True)
    
    print("\n--- Top Candidates Ranked by Overall Sequence Probability (PLL) ---")
    for i, cand in enumerate(candidates):
        print(f"Rank {i+1} \t\t\t| PLL Score: {cand['pll']:>7.2f} | Linker: {cand['linker']}")

    greedy_linker, greedy_sequence = design_linker_iterative(domain_a, 
                                                             domain_b, 
                                                             linker_length, 
                                                             model=model, 
                                                             tokenizer=tokenizer)
    
    pll_score_greedy = get_sequence_pll(greedy_sequence, linker_start_idx, linker_length, model, tokenizer)
    exp_linker = 'SGGGGGGS'
    experimental_baseline = get_sequence_pll(domain_a+exp_linker+domain_b, linker_start_idx, len(exp_linker), model, tokenizer)

    print(f"Greedy Baseline \t| PLL Score: {pll_score_greedy:>7.2f} | Linker: {greedy_linker}")
    print(f"Experiment Baseline \t| PLL Score: {experimental_baseline:>7.2f} | Linker: {exp_linker}")

    return candidates

def read_fasta(filepath):
    """
    Reads a FASTA file and returns a list of sequences.
    """
    sequences = []
    current_seq = []
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                elif line:
                    # Remove any whitespace or hidden characters from sequence lines
                    current_seq.append(line.replace(" ", ""))
            if current_seq:
                sequences.append("".join(current_seq))
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        sys.exit(1)
        
    return sequences

def design_linker(domain_a, domain_b, linker_length, model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Uses Meta's ESM-2 model to predict a linker sequence between two protein domains.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()

    masked_linker = "<mask>" * linker_length
    full_sequence = domain_a + masked_linker + domain_b
    
    print(f"\nInput Sequence (with {linker_length} masks):")
    print(full_sequence)

    inputs = tokenizer(full_sequence, return_tensors="pt")
    mask_token_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    print("\nPredicting linker sequence...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_linker = ""
    for idx in mask_token_indices:
        mask_logits = logits[0, idx, :]
        top_token_id = torch.argmax(mask_logits, dim=-1).item()
        predicted_aa = tokenizer.decode([top_token_id])
        predicted_linker += predicted_aa.strip()

    final_protein = domain_a + predicted_linker + domain_b
    
    print(f"\n--- Results ---")
    print(f"Predicted Linker: {predicted_linker}")
    print(f"Final Fusion Protein:\n{final_protein}\n")
    
    return predicted_linker, final_protein

def design_linker_iterative(domain_a, domain_b, linker_length, 
                            model_name="facebook/esm2_t6_8M_UR50D",
                            silent=True, model=None, tokenizer=None):
    """
    Uses Meta's ESM-2 model to iteratively predict a linker sequence.
    It fills the most confident mask one at a time, allowing local dependencies to guide the folding.
    """
    if model is None or tokenizer is None:
        if not silent:
            print(f"Loading model: {model_name}...")
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
        model.eval()

    # Represent the linker as a list so we can update individual positions
    current_linker = ["<mask>"] * linker_length
    if not silent:
        print(f"\nStarting iterative decoding for {linker_length} masks...")

    # We need to do exactly as many passes as there are masks
    for step in range(linker_length):
        # 1. Construct the current sequence
        full_sequence = domain_a + "".join(current_linker) + domain_b
        inputs = tokenizer(full_sequence, return_tensors="pt")
        
        # 2. Pass through the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # 3. Find where the masks currently are in the tokenized tensor
        mask_token_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        
        best_confidence = -1.0
        best_token_id = -1
        linker_pos_to_update = -1

        # Track the actual positions in our current_linker list that still need filling
        remaining_mask_positions = [idx for idx, val in enumerate(current_linker) if val == "<mask>"]
        
        # 4. Evaluate confidence for every remaining mask
        for i, tensor_idx in enumerate(mask_token_indices):
            # Convert raw logits to probabilities
            probs = F.softmax(logits[0, tensor_idx, :], dim=-1)
            
            # Find the highest probability token for this specific mask
            confidence, token_id = torch.max(probs, dim=-1)
            
            # If this mask's top prediction is more confident than our previous best, save it
            if confidence.item() > best_confidence:
                best_confidence = confidence.item()
                best_token_id = token_id.item()
                linker_pos_to_update = remaining_mask_positions[i]
        
        # 5. Decode the winning token and update that single position in our linker list
        predicted_aa = tokenizer.decode([best_token_id]).strip()
        current_linker[linker_pos_to_update] = predicted_aa
        if not silent:
            print(f"Step {step+1}: Filled position {linker_pos_to_update+1} with '{predicted_aa}' (Confidence: {best_confidence:.4f})")
            print(f"Current Linker: {''.join(current_linker)}")

    # 6. Finalize the protein
    predicted_linker = "".join(current_linker)
    final_protein = domain_a + predicted_linker + domain_b

    if not silent:
        print(f"\n--- Final Results ---")
        print(f"Predicted Linker: {predicted_linker}")
        print(f"Final Fusion Protein:\n{final_protein}\n")
    
    return predicted_linker, final_protein

def main():
    parser = argparse.ArgumentParser(
        description="Design a linker between two protein domains using an LLM (ESM-2)."
    )
    
    # Existing arguments
    parser.add_argument("-s1", type=str, help="N-terminal sequence (Domain A)")
    parser.add_argument("-s2", type=str, help="C-terminal sequence (Domain B)")
    parser.add_argument("-l", "--length", type=int, default=5, help="Length of the linker")
    parser.add_argument("-f", "--fasta", type=str, help="Path to a FASTA file")
    parser.add_argument("-i", "--inverse", action="store_true", help="Make c-terminal fusion instead")
    parser.add_argument("-m", "--model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--method", type=str, choices=['greedy', 'iterative', 'gibbs'], default='greedy',
                        help="Decoding method: 'greedy', 'iterative', or 'gibbs'.")
    parser.add_argument("-n", "--num_seqs", type=int, default=10, 
                        help="Number of sequences to generate (only for gibbs)")
    parser.add_argument("--masks", type=int, default=2, 
                        help="Number of residues to mask per step (only for gibbs)")
    parser.add_argument("--steps", type=int, default=50, 
                        help="Number of Gibbs iterations per sequence (only for gibbs)")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, 
                        help="Sampling temperature (1.0 is standard, >1.0 more random, <1.0 more strict)")

    args = parser.parse_args()

    # Input validation logic (same as before)
    if args.fasta:
        sequences = read_fasta(args.fasta)
        if len(sequences) < 2:
            print("Error: The FASTA file must contain at least two sequences.")
            sys.exit(1)
        domain_a = sequences[0]
        domain_b = sequences[1]
    elif args.s1 and args.s2:
        domain_a = args.s1
        domain_b = args.s2
    else:
        print("Error: Provide either -s1/-s2, or -f/--fasta file.")
        sys.exit(1)

    if args.inverse:
        domain_a, domain_b = domain_b, domain_a

    # Execution routing
    if args.method == 'gibbs':
        print("\nUsing GIBBS SAMPLING decoding method.")
        design_linker_gibbs(domain_a, domain_b, args.length, 
                            num_seqs=args.num_seqs, masks_per_step=args.masks, 
                            steps=args.steps, temperature=args.temperature, model_name=args.model)
    elif args.method == 'iterative':
        print("\nUsing ITERATIVE decoding method.")
        design_linker_iterative(domain_a, domain_b, args.length, model_name=args.model)
    else:
        print("\nUsing GREEDY decoding method.")
        design_linker(domain_a, domain_b, args.length, model_name=args.model)

if __name__ == "__main__":
    main()
