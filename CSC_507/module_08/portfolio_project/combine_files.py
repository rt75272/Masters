import os
import multiprocessing
# ---------------------------------------------------------------------------------------
# Large Data Processor.
#
# Reads one billion lines from each input file, adds each line together, and save to a 
# new output file. 
#
# Usage:
#	$ python combine_files.py
# ---------------------------------------------------------------------------------------
"""
Splits input file(containing 1 billion lines/words) into 'num_parts' smaller files ensuring enough lines.
"""
def split_file(input_file, total_lines_needed=1000000000, num_parts=10):
    # Get the total number of lines in the input file.
    with open(input_file, 'r') as infile:
        total_lines = sum(1 for line in infile)  # Count total number of lines in the file.
    # Check if the input file has enough lines.
    if total_lines < total_lines_needed:
        raise ValueError(f"Input file does not have enough lines to produce {total_lines_needed} lines")
    # Determine how many lines each part should have.
    lines_per_file = total_lines_needed // num_parts  # Divide lines equally for the desired output size.
    # Open the file again to start reading and splitting it into parts.
    with open(input_file, 'r') as infile:
        for part_num in range(num_parts):  # Iterate through each part.
            output_file = f"{input_file}_part{part_num + 1}.txt"
            with open(output_file, 'a+') as outfile:
                for _ in range(lines_per_file):
                    line_content = infile.readline()
                    if line_content == '':  # If the end of the file is reached.
                        break
                    outfile.write(line_content)  # Write the line to the output part file.
                # If it's the last part, write the remaining lines to this file.
                if part_num == num_parts - 1:
                    for line_content in infile:
                        outfile.write(line_content)

"""
Process a chunk of two files, summing corresponding lines.
"""
def process_chunk(file1_name, file2_name, start_line, lines_to_process):
    results = []  # To store the sum of the corresponding lines.
    with open(file1_name, 'r') as f1, open(file2_name, 'r') as f2:
        # Skip lines until we reach the starting point of the chunk.
        for i in range(start_line):
            f1.readline()  # Skip 'start_line' lines in file1.
            f2.readline()  # Skip 'start_line' lines in file2.
        # Process 'lines_to_process' lines from both files.
        for _ in range(lines_to_process):
            line1 = f1.readline().strip()  # Read a line from file1 and remove trailing whitespaces.
            line2 = f2.readline().strip()  # Read a line from file2 and remove trailing whitespaces.
            if not line1 or not line2:
                break  # If we've reached the end of either file, stop processing.
            num1, num2 = map(int, (line1, line2))  # Convert both lines to integers.
            total = num1 + num2  # Sum the corresponding numbers.
            results.append(f"{total}\n")  # Store the result.
    return results  # Return the list of summed results.

"""
Process file chunks in parallel using multiple processes.
"""
def process_file_chunks(file1_parts, file2_parts, num_processes=10):
    # Create a pool of worker processes to process chunks in parallel.
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []  # List to store results of each parallel process.
        # Calculate the number of lines to process per chunk.
        lines_per_chunk = sum(1 for line in open(file1_parts[0])) // len(file1_parts)  # Lines per part.
        # For each pair of file parts (one from file1, one from file2).
        for i in range(len(file1_parts)):
            start_line = i * lines_per_chunk  # Calculate the starting line for each chunk.
            # Submit the task to the multiprocessing pool to process this chunk.
            results.append(pool.apply_async(process_chunk, (file1_parts[i], file2_parts[i], start_line, lines_per_chunk)))
        # After all chunks are processed, collect the results and write them to the output file.
        with open('totalfile.txt', 'a+') as total_file:
            for result in results:
                total_file.writelines(result.get())  # Collect and write the results to the output file.
        print("Processing completed and results written to totalfile.txt.")

"""
Main driver function.
"""
def main():
    # Split the huge files into parts for parallel processing.
    split_file('hugefile1.txt', total_lines_needed=1000000000, num_parts=10)
    split_file('hugefile2.txt', total_lines_needed=1000000000, num_parts=10)
    # Generate a list of part filenames for both input files.
    file1_parts = [f"hugefile1.txt_part{i+1}.txt" for i in range(10)]
    file2_parts = [f"hugefile2.txt_part{i+1}.txt" for i in range(10)]
    # Process these parts in parallel using multiprocessing.
    process_file_chunks(file1_parts, file2_parts)
    # Clean up by removing the temporary part files after processing.
    for part in file1_parts + file2_parts:
        os.remove(part)  # Delete the part files to free up space.
    print("Temporary files removed.")

# Big red activation button.
if __name__ == '__main__':
    main()