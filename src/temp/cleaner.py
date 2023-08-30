import csv

def remove_newlines(input_str):
    return input_str.replace('\n', ' ')

input_file = r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db5-Zinc-250k\inputs.csv'
output_file = r'C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db5-Zinc-250k\new_inputs.csv'

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        cleaned_row = [remove_newlines(cell) for cell in row]
        writer.writerow(cleaned_row)

print("Newline removal completed.")