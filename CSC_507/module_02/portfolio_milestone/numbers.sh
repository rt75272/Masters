# ---------------------------------------------------------------------------------------
# BASH Number Generator.
#
# Generates a thousand random integers and saves them to a file named file1.txt.
#
# Usage:
#	$ chmod +x numbers.sh
#	$ ./numbers.sh
# ---------------------------------------------------------------------------------------
# Initial setup.
n=1000 # Upper limit.
filename=file1.txt # BASH output file.

# Main driver function.
main() {
	# Loop from one to one thousand, incrementing by one.
	for i in $(seq 1 $n) # Sequence from one to n. Replaces {1..1000}.
	do
		echo "$RANDOM" >> $filename # Generate a random integer and append to file1.txt.
	done
}

# Big red activation button.
main

