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
	temp_file="temp_$1.txt"

    # Using /dev/urandom to generate faster random numbers.
    # od converts random bytes into integers (1 per line), and we limit output to exactly $n_per_core numbers.
    head -c $((n_per_core * 16)) /dev/urandom | od -An -N$((n_per_core * 16)) | awk '{print $1}' > "$temp_file"
    # echo "Thread $1 completed."
}

# Displays the looper's runtime.
printer() {
	runtime=$1 # Assign first parameter to runtime.
	runtime=$(printf "%8.4f\n" "$runtime") # Convert to four decimal places.
	echo "Bash runtime:" $runtime" seconds"
}

# Split looper's processing up into the number of available cpu cores.
multithread() {
	for i in $(seq 1 $num_cores); do
		looper $i &
	done
	wait
    cat temp_*.txt > "$filename"
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
