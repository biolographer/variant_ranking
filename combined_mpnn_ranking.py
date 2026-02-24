import subprocess
import numpy as np
import pandas as pd
import torch
import os
from transformers import AutoTokenizer, EsmForMaskedLM

MPNN_PATH = '/Users/aarondebon/bioinformatics/ProteinMPNN/'
RANK_PATH = '/Users/aarondebon/bioinformatics/variant_ranking'

# --- 1. Run Structure-Based Model (Original ProteinMPNN) ---
def get_mpnn_scores(pdb_path, output_dir=f"{RANK_PATH}/mpnn_dms_out", mpnn_dir=MPNN_PATH):
    # 1. Use the correct flag for a zero-shot structure DMS
    command = [
        "python", "protein_mpnn_run.py", 
        "--pdb_path", os.path.abspath(pdb_path),
        "--out_folder", os.path.abspath(output_dir),
        "--conditional_probs_only", "1",
        "--conditional_probs_only_backbone", "1", 
        "--batch_size", "1"
    ]

    print(f"\n--- DEBUG INFO ---")
    print(f"Target PDB: {os.path.abspath(pdb_path)}")
    print(f"MPNN Dir:   {mpnn_dir}")
    print(f"------------------\n")

    try:
        # We use capture_output=True instead of DEVNULL to catch the hidden errors
        result = subprocess.run(
            command, 
            check=True, 
            cwd=mpnn_dir, 
            capture_output=True,
            text=True
        )
        print("ProteinMPNN Output:")
        print(result.stdout)
        if result.stderr:
            print("ProteinMPNN Internal Warnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print("ProteinMPNN crashed with a system error!")
        print(e.stderr)
        raise

    # 2. Locate the generated .npz file
    # The backbone flag saves to a specific subfolder
    npz_folder = os.path.join(output_dir, "conditional_probs_only")
    npz_files = [f for f in os.listdir(npz_folder) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError("ProteinMPNN did not output the expected .npz file.")
        
    # 3. Load the data
    data = np.load(os.path.join(npz_folder, npz_files[0]))
    
    # Under this flag, the keys are 'log_p' and 'S'
    log_p = data['log_p'].squeeze()  # Shape: (Length, 21)
    S = data['S'].squeeze()          # The TRUE Wild-Type sequence parsed from the PDB!
    
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    records = []
    
    # 4. Calculate Δ log p
    for pos_idx in range(len(S)):
        wt_idx = S[pos_idx]
        wt_aa = alphabet[wt_idx]
        
        # Skip gaps or unknown residues
        if wt_aa == 'X': 
            continue 
            
        wt_log_prob = log_p[pos_idx, wt_idx]
        
        # Calculate Δ log p for the 20 standard amino acids
        for mut_idx, mut_aa in enumerate(alphabet[:-1]): 
            mut_log_prob = log_p[pos_idx, mut_idx]
            
            # Positive score = predicted stabilizing mutation
            score = mut_log_prob - wt_log_prob
            
            records.append({
                'pos': pos_idx + 1,
                'pre': wt_aa,
                'post': mut_aa,
                'mpnn_score': score
            })
            
    return pd.DataFrame(records)

# --- 2. Run Sequence-Based Model (ESM-2) ---
def get_esm_scores(df_mpnn):
    print("Running ESM-2 for evolutionary probabilities...")
    
    # Reconstruct the Wild-Type sequence from the MPNN DataFrame
    wt_residues = df_mpnn.drop_duplicates('pos').sort_values('pos')['pre'].tolist()
    wt_seq = "".join(wt_residues)
    print(f'\n_________________\nSequence:\n{wt_seq}\n_________________')
    
    # Load ESM-2
    model_name = "facebook/esm2_t6_8M_UR50D" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    records = []

    for i, wt_aa in enumerate(wt_seq):
        masked_seq = wt_seq[:i] + "<mask>" + wt_seq[i+1:]
        inputs = tokenizer(masked_seq, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**inputs).logits
            
        mask_idx = i + 1 # Offset by 1 because of the <cls> token at the start
        pos_logits = logits[0, mask_idx, :]
        
        wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
        wt_logit = pos_logits[wt_token_id].item()
        
        for mut_aa in amino_acids:
            mut_token_id = tokenizer.convert_tokens_to_ids(mut_aa)
            mut_logit = pos_logits[mut_token_id].item()
            
            score = mut_logit - wt_logit 
            records.append({'pos': i + 1, 'pre': wt_aa, 'post': mut_aa, 'esm_score': score})
            
    return pd.DataFrame(records)

# --- 3. Combine and Normalize ---
def generate_combined_dms(pdb_path):
    # Get scores from both models
    df_mpnn = get_mpnn_scores(pdb_path)
    df_esm = get_esm_scores(df_mpnn)
    
    # Merge the dataframes
    print("Merging and calculating Z-scores...")
    df_merged = pd.merge(df_mpnn, df_esm, on=['pos', 'pre', 'post'])
    
    # Calculate Z-Scores: Z = (Score - Mean) / Standard_Deviation
    df_merged['mpnn_z'] = (df_merged['mpnn_score'] - df_merged['mpnn_score'].mean()) / df_merged['mpnn_score'].std()
    df_merged['esm_z'] = (df_merged['esm_score'] - df_merged['esm_score'].mean()) / df_merged['esm_score'].std()
    
    # Combine Z-scores
    df_merged['combined_z'] = df_merged['mpnn_z'] + df_merged['esm_z']
    
    # Rank by the best combined score
    df_ranked = df_merged.sort_values('combined_z', ascending=False).reset_index(drop=True)
    
    return df_ranked

# --- Execute ---
if __name__ == "__main__":

    pdb_file = "/Users/aarondebon/bioinformatics/variant_ranking/data/5OD1.pdb" 
    
    final_df = generate_combined_dms(pdb_file)
    
    print("\n--- Top 10 Combined Mutations ---")
    print(final_df.head(10).to_string(index=False))

    if not os.path.exists('./output'):
        os.makedirs('./output')

    final_df.to_csv("./output/combined_dms_ranked_base_mpnn.csv", index=False)
    print("\nResults saved to 'combined_dms_ranked_base_mpnn.csv'")