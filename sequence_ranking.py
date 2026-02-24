import torch
import esm
import pandas as pd

def score_all_mutations(model_name, wt_sequence, grouping_strategy="substitution"):
    """
    Runs a full zero-shot deep mutational scan using wild-type marginals 
    and applies MULTI-evolve Z-score normalization.
    """
    print(f"Loading {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    # The standard 20 amino acids
    standard_aas = list("ACDEFGHIKLMNPQRSTVWY")
    
    # 1. Prepare data and run the SINGLE forward pass
    data = [("protein", wt_sequence)]
    _, _, batch_tokens = batch_converter(data)
    
    print("Running forward pass...")
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
        
    # Logits for the first (and only) sequence in the batch
    logits = results["logits"][0]
    
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # 2. Extract Fold-Change (FC) scores for all possible mutations
    print("Calculating Fold-Change scores...")
    mutation_data = []
    
    # Iterate through the sequence (1-indexed to match typical biology notation)
    # Note: token 0 is <cls>, so the sequence starts at index 1 in the tensor
    for pos in range(1, len(wt_sequence) + 1):
        wt_aa = wt_sequence[pos - 1]
        wt_idx = alphabet.get_idx(wt_aa)
        
        # Get the wild-type likelihood p(x_i | x)
        wt_prob = probabilities[pos, wt_idx].item()
        
        for mut_aa in standard_aas:
            if mut_aa == wt_aa:
                continue # Skip synonymous mutations (e.g., A -> A)
                
            mut_idx = alphabet.get_idx(mut_aa)
            
            # Get the mutated likelihood p(x'_i | x)
            mut_prob = probabilities[pos, mut_idx].item()
            
            # Calculate Fold-Change
            fc_score = mut_prob / wt_prob
            
            mutation_data.append({
                'position': pos,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'mutation': f"{wt_aa}{pos}{mut_aa}",
                'fc_score': fc_score
            })
            
    df = pd.DataFrame(mutation_data)
    
    # 3. Apply MULTI-evolve Z-Score Normalization
    print(f"Applying Z-score normalization (Strategy: {grouping_strategy})...")
    
    if grouping_strategy == "substitution":
        df['group_key'] = df['wt_aa'] + "->" + df['mut_aa']
    elif grouping_strategy == "target":
        df['group_key'] = "->" + df['mut_aa']
    else:
        raise ValueError("Invalid grouping_strategy. Use 'substitution' or 'target'.")
        
    # Calculate mu_S and sigma_S
    group_stats = df.groupby('group_key')['fc_score'].agg(['mean', 'std']).reset_index()
    group_stats.rename(columns={'mean': 'mu_S', 'std': 'sigma_S'}, inplace=True)
    
    df = df.merge(group_stats, on='group_key', how='left')
    
    # Handle edge cases where std is NaN or 0
    df['sigma_S'] = df['sigma_S'].fillna(1.0)
    df.loc[df['sigma_S'] == 0, 'sigma_S'] = 1.0
    
    # Calculate Z-score
    df['z_score'] = (df['fc_score'] - df['mu_S']) / df['sigma_S']
    
    # Sort by Z-score descending to rank the best candidates
    df = df.sort_values(by='z_score', ascending=False).reset_index(drop=True)
    
    print("Done!")
    return df

# --- Example Usage ---
if __name__ == "__main__":
    # Use a small ESM model for testing; swap to 'esm2_t36_3B_UR50D' for production
    MODEL_ID = "esm2_t33_650M_UR50D" 
    
    # A short dummy sequence for fast execution
    SEQUENCE = "MKVLYYGRTLAE" 
    
    # Run the pipeline grouping by exact substitution 
    results_df = score_all_mutations(MODEL_ID, SEQUENCE, grouping_strategy="substitution")
    
    print("\n--- Top 10 Predicted Mutations ---")
    # Display the most relevant columns
    print(results_df[['mutation', 'group_key', 'fc_score', 'z_score']].head(10))
