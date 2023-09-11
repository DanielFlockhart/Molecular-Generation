import csv


path = r"C:\Users\0xdan\Documents\CS\WorkCareer\Chemistry Internship\Project-Code\data\datasets\db6-pas-combined"


# Specify the input and output file paths
input_file_path = path + "\data.txt"
output_file_path =  path + "\output.txt"
# Create a set to store unique lines
unique_lines = set()

# Open the input file for reading and the output file for writing
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Iterate through each line in the input file
    for line in input_file:
        # Strip leading whitespace from the line and remove trailing newline characters
        stripped_line = line.lstrip().rstrip('\n')
        
        # Add the stripped line to the set if it's not already present (removing duplicates)
        unique_lines.add(stripped_line)

    # Write the unique lines to the output file
    for line in unique_lines:
        output_file.write(line + '\n')

print("Duplicate lines removed. Output saved to", output_file_path)