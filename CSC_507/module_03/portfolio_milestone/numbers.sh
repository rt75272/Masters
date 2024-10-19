# ---------------------------------------------------------------------------------------
# BASH Number Generator.
#
# Generates a million random integers and saves them to a file named file1.txt. Also, 
# calculates and displays the runtime.
#
# Usage:
#	$ chmod +x numbers.sh
#	$ ./numbers.sh
# ---------------------------------------------------------------------------------------
# Initial setup.
n=1000000 # Upper limit.
num_cores=$(nproc --all) # Get the number of cpu cores available.
n_per_core=$(($n / $num_cores))  # Divide n up into the number of cores.
filename=file1.txt # BASH output file.

# Generates n random integers and adds to file.
looper() {
	temp_file="temp_$1.txt" # Temp file for storage.
	jot -r $n_per_core > "$temp_file" # Generate n_per_core random numbers and store in temp_file.
}

# Displays the looper's runtime.
printer() {
	runtime=$1 # Assign first parameter to runtime.
	runtime=$(printf "%8.4f\n" "$runtime") # Convert to four decimal places.
	echo "Bash runtime:" $runtime" seconds"
}

# Split looper's processing up into the number of available cpu cores.
multithread() {
	# Loop through the amount of cores and give each a looper task.
	for i in $(seq 1 $num_cores); do
		looper $i & # Run looper in its own thread.
	done
	wait # Wait for all threads to complete execution.
    cat temp_*.txt > "$filename" # Move temp files content into a single output file.
}

# Main driver function.
main() {
	start_time=$(date +"%s.%N") # Start runtime timer.
	multithread # Run several threads to produce the million random integers faster.
	end_time=$(date +"%s.%N") # End runtime timer.
	duration=$(echo "$end_time - $start_time" | bc) # Subtract times and store the result.
	printer $duration # Final results.
}

# Big red activation button.
main
