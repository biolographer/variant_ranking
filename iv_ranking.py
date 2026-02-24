import subprocess
import numpy as np
import os

def generate_mpnn_dms(pdb_path, output_dir):
    # 1. Call ProteinMPNN to dump the backbone probabilities
    command = [
        "python", "./protein_mpnn_run.py",
        "--pdb_path", pdb_path,
        "--out_folder", output_dir,
        "--conditional_probs_only_backbone", "1", # The magic flag
        "--batch_size", "1"
    ]
    
    print("Extracting probabilities from ProteinMPNN...")
    subprocess.run(command, check=True)
    
    # 2. Locate and load the resulting .npz file
    # ProteinMPNN saves this inside a specific subfolder
    npz_folder = os.path.join(output_dir, "conditional_probs_only")
    npz_file = [f for f in os.listdir(npz_folder) if f.endswith(".npz")][0]
    data = np.load(os.path.join(npz_folder, npz_file))
    
    # Tip: Print data.files to see all available arrays (e.g., 'log_p', 'S', etc.)
    # log_p shape is usually (Length, 21) -> 20 standard amino acids + 1 gap/unknown
    log_probs = data['log_p'] 
    
    # 'S' is the input sequence encoded as integer indices (0-20)
    wt_sequence_indices = data['S'] 
    
    # 3. Calculate the DMS score matrix: mutant - wildtype
    dms_matrix = np.zeros_like(log_probs)
    
    for position in range(len(wt_sequence_indices)):
        wt_index = wt_sequence_indices[position]
        wt_log_prob = log_probs[position, wt_index]
        
        # Calculate Δ log p for all 20 AAs at this position
        # Positive value = predicted stabilizing; Negative = destabilizing
        dms_matrix[position, :] = log_probs[position, :] - wt_log_prob
        
    return dms_matrix, data.files

# Run the function
dms_scores, available_keys = generate_mpnn_dms("your_protein.pdb", "./dms_results")
print("DMS Matrix Shape:", dms_scores.shape)