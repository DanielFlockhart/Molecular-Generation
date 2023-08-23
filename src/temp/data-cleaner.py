from chembl_webresource_client.new_client import new_client
from tqdm import tqdm  # Import tqdm

client = new_client

# Open the input file for reading
input_file_path = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\CHEMBL_IDS.txt"
output_file_path = "outs.txt"

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    num_lines = sum(1 for line in input_file)  # Count the total lines
    input_file.seek(0)  # Reset the file pointer

    for i, line in enumerate(tqdm(input_file, total=num_lines)):  # Use tqdm for progress bar
        try:
            molecule = client.molecule.get(line.strip())  # Remove leading/trailing whitespace
            canonical_smiles = molecule["molecule_structures"]["canonical_smiles"]
            output_file.write(canonical_smiles + "\n")
        except:
            print(f"Error processing line {i + 1}")


# Finished processing
print("Processing complete.")
