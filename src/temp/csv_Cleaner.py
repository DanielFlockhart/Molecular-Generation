
file = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db4-testing\dataset.csv"

import csv

def remove_all_whitespace(s):
    return ''.join(s.split())

input_csv_path = file

# Read the CSV file and clean the values in place
with open(input_csv_path, 'r', newline='', encoding='utf-8') as input_file:
    csv_reader = csv.reader(input_file)
    data = list(csv_reader)  # Read all rows into a list

# Clean the values in the data
for row_idx, row in enumerate(data):
    for col_idx, cell in enumerate(row):
        cleaned_cell = remove_all_whitespace(cell)  # Remove all whitespace characters
        data[row_idx][col_idx] = cleaned_cell

# Overwrite the original CSV file with the cleaned values
with open(input_csv_path, 'w', newline='', encoding='utf-8') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerows(data)