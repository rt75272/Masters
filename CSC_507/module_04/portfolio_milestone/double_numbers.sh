#!/bin/bash
# ---------------------------------------------------------------------------------------
# Double Numbers Script.
#
# Reads numbers from file1.txt, doubles each number, and writes the result to newfile1.txt.
#
# Usage:
#   $ chmod +x double_numbers.sh
#   $ ./double_numbers.sh
# ---------------------------------------------------------------------------------------
input_file="file1.txt"   # Input file with random numbers.
output_file="newfile1.txt" # Output file where doubled numbers will be saved.

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: $input_file does not exist."
    exit 1
fi

# Read each line from file1.txt, double the number, and write it to newfile1.txt
while IFS= read -r number; do
    doubled_number=$((number * 2))  # Double the number
    echo "$doubled_number" >> "$output_file"  # Append the doubled number to the output file
done < "$input_file"

echo "Doubled numbers have been saved to $output_file."
