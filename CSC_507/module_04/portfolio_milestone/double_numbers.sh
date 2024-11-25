#!/bin/bash
# ---------------------------------------------------------------------------------------
# Double Numbers Script
#
# Reads numbers from file1.txt, doubles each number, and writes the result to 
# newfile1.txt.
#
# Usage:
#   $ chmod +x double_numbers.sh
#   $ ./double_numbers.sh
# ---------------------------------------------------------------------------------------
input_file="file1.txt"   # Input file with random numbers.
output_file="newfile1.txt" # Output file where doubled numbers will be saved.

# Check if input file exists.
if [ ! -f "$input_file" ]; then
    echo "Error: $input_file does not exist."
    exit 1
fi

# Ensure the output file is writable, if it exists.
if [ -f "$output_file" ] && [ ! -w "$output_file" ]; then
    echo "Error: $output_file exists but is not writable."
    exit 1
fi

start_time=$(date +%s.%N) # Capture the start time with nanosecond precision.
awk '{print $1 * 2}' "$input_file" > "$output_file" # Read and double number, write to output file.

# Check if awk command was successful.
if [ $? -eq 0 ]; then
    echo "Doubled numbers have been saved to $output_file."
else
    echo "Error: Failed to write to $output_file."
    exit 1
fi

end_time=$(date +%s.%N) # Capture the end time with nanosecond precision.
runtime=$(echo "$end_time - $start_time" | bc) # Calculate the runtime using `bc` for precision.
echo "Script execution time: $runtime seconds." # Display the runtime with decimal places.
