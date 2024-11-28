import os
import multiprocessing
import sys
# ---------------------------------------------------------------------------------------
# Python File Splitter
#
# Takes in an input file and breaks it up into smaller parts/files, processes the data, 
# then combines all the parts into a single final output file.
#
# Usage:
#	$ python splitter.py <input_file> <num_parts>
# ---------------------------------------------------------------------------------------
# Split the main input file into n smaller files.
def split_file(input_file, num_parts):
    # Read the entire input file.
    with open(input_file, 'r') as file:
        lines = file.readlines()
    lines_per_file = len(lines) // num_parts # Determine the number of lines per file.
    # Split and write to smaller files.
    file_parts = []
    for i in range(num_parts):
        part_filename = f"file_part_{i+1}.txt"
        start_line = i * lines_per_file
        end_line = (i + 1) * lines_per_file if i < num_parts - 1 else len(lines)
        with open(part_filename, 'w') as part_file:
            part_file.writelines(lines[start_line:end_line])
        file_parts.append(part_filename)
    return file_parts

# Processing for each file.
def process_file(file_part):
    output_file = f"processed_{file_part}"
    # Read the file's contents.
    with open(file_part, 'r') as file:
        lines = file.readlines()
    # Processing all the file's contents.
    with open(output_file, 'w') as file:
        file.writelines(lines)
    return output_file

# Combine all processed files into a single output file.
def combine_files(file_parts, output_file):
    with open(output_file, 'w') as outfile:
        for part in file_parts:
            with open(part, 'r') as infile:
                outfile.writelines(infile.readlines())

# Main driver function.
def main(input_file, num_processes, output_file="final_output.txt"):
    file_parts = split_file(input_file, num_processes) # Split the file into n parts.
    # Process each part in parallel.
    with multiprocessing.Pool(processes=num_processes) as pool:
        processed_files = pool.map(process_file, file_parts)
    combine_files(processed_files, output_file) # Combine the processed files into one.
    print(f"Processing complete. Output written to {output_file}.")

# Big red activation button.
if __name__ == "__main__":
    # Ensure that the user provides the correct number of arguments.
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python splitter.py <num_processes> [input_file]")
        sys.exit(1)
    # Process desired number of processes argument.
    try:
        num_processes = int(sys.argv[1])  # Number of processes.
    except ValueError:
        print("Error: num_processes must be an integer.")
        sys.exit(1)
    # If the user provided an input file, use it; otherwise, default to "file2.txt".
    input_file = None
    if len(sys.argv) == 3:
        input_file = sys.argv[2] 
    else:
        "file2.txt"
    # Check if the input file exists.
    if not os.path.exists(input_file):
        print(f"Error: The file {input_file} does not exist.")
        sys.exit(1)
    # Run the main function.
    main(input_file, num_processes)

   