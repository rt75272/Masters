import os
import time
# ---------------------------------------------------------------------------------------
# Python Numbers File Processor
#
# Reads the entire contents of file1.txt into memory, then process each row. Reads one 
# row of file1.txt at a time and processes it. Splits file1.txt into 2 parts and reads each
# part into memory separately.
#
# Usage:
#   $ python processor.py
# ---------------------------------------------------------------------------------------

"""
Doubles numbers and writes them to a specified output file.
"""
def double_numbers(numbers, output_file):
    doubled_numbers = []
    # Only process non-empty numbers and write the doubled numbers to the output file.
    for number in numbers:
        stripped_number = number.strip()
        if stripped_number.isdigit():
            doubled_numbers.append(str(int(stripped_number) * 2))
    # Write all doubled numbers to the output file.
    with open(output_file, 'w') as out_file:
        out_file.write("\n".join(doubled_numbers))

"""
Reads entire file contents into memory.
"""
def read_all_into_memory(input_file, output_file):
    start_time = time.time()
    # Read all lines from the input file into memory.
    with open(input_file, 'r') as file:
        lines = file.readlines()
    double_numbers(lines, output_file)  # Process the numbers and write them to the output file.
    end_time = time.time()
    print(f"Method read_all_into_memory() execution time: {end_time - start_time:.4f} seconds.")

"""
Reads one row of the file at a time and processes it.
"""
def read_row_by_row(input_file, output_file):
    start_time = time.time()
    doubled_numbers = []
    # Read the input file line by line and process it.
    with open(input_file, 'r') as in_file:
        for line in in_file:
            # Check if line is a valid number.
            if line.strip().isdigit():
                doubled_number = str(int(line.strip()) * 2)
                doubled_numbers.append(doubled_number)
    # Write the results to the output file.
    with open(output_file, 'w') as out_file:
        out_file.write("\n".join(doubled_numbers))
    end_time = time.time()
    print(f"Method read_row_by_row() execution time: {end_time - start_time:.4f} seconds.")

"""
Splits the file into two parts and processes each part separately into memory.
"""
def halve_file_into_memory(input_file, output_file):
    start_time = time.time()
    file_size = os.path.getsize(input_file) # Get the size of the input file.
    half_size = file_size // 2
    # Read the first half of the file.
    with open(input_file, 'r') as file:
        part1 = file.read(half_size).splitlines()
        part2 = file.read().splitlines()
    doubled_numbers = []
    # Process the first half.
    for number in part1:
        if number.strip().isdigit():
            doubled_numbers.append(str(int(number) * 2))
    # Process the second half.
    for number in part2:
        if number.strip().isdigit():
            doubled_numbers.append(str(int(number) * 2))
    # Write the results to the output file.
    with open(output_file, 'w') as out_file:
        out_file.write("\n".join(doubled_numbers))  
    end_time = time.time()
    print(f"Method halve_file_into_memory() execution time: {end_time - start_time:.4f} seconds.")

"""
Main driver function.
"""
def main():
    input_file = 'file1.txt'
    output_file = 'newfile1.txt'
    read_all_into_memory(input_file, output_file) # Read all into memory and process.
    read_row_by_row(input_file, output_file) # Read line by line and process.
    halve_file_into_memory(input_file, output_file) # Halve file and process each part.

# Big red activation button.
if __name__ == "__main__":
    main()
