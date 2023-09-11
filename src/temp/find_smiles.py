import pubchempy as pcp
from tqdm import tqdm  # Import tqdm for the progress bar

data = []
path = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db6-pas-combined\smiles.txt"

# Count the number of lines in the file to determine the total iterations for tqdm
with open(path, "r+", encoding="utf8") as f:
    num_lines = sum(1 for line in f)

with open(path, "r+", encoding="utf8") as f:
    for row in tqdm(f.read().split("\n"), total=num_lines, desc="Processing"):
        try:
            results = pcp.get_compounds(row, 'name')
            for res in results:
                data.append([row, res.canonical_smiles])
        except Exception as e:
            print(f"Error {e}")

# Save the data to a file
with open("model_train.txt", "w+", encoding="utf8") as f:
    f.write(str(data))
