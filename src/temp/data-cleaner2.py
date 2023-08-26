    
file = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db5-Zinc-250k\dataset.csv"
file2= r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db5-Zinc-250k\new_dataset.csv"
import csv

# Open the input CSV file for reading and the output CSV file for writing
with open(file, 'r', newline='') as input_file, open(file2, 'w', newline='') as output_file:
    csv_reader = csv.DictReader(input_file)
    fieldnames = csv_reader.fieldnames
    
    # Create a CSV writer with the same fieldnames as the input
    csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    
    for row in csv_reader:
        # Assuming "SMILES" is the column header containing SMILES strings
        smiles = row['SMILES']
        cleaned_smiles = smiles.replace('\n', '')
        row['SMILES'] = cleaned_smiles
        
        csv_writer.writerow(row)

print("CSV processing complete.")
