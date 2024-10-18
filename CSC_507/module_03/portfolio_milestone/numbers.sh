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
num_cores=$(nproc --all)
n=$(($n/$num_cores))
# echo $n | figlet
filename=file1.txt # BASH output file.

# Generates n random integers and adds to file.
looper() {
	# Loop from one to one million, incrementing by one.
	for i in $(seq 1 $n); do # Sequence from one to n. Replaces {1..1000}.
		echo "$RANDOM" # Generate a random integer.
	done >> "$filename" # Append to file all at once, at the end. 
}

# Displays the looper's runtime.
printer() {
	x=$1 # Assign first parameter to x.
	x=$(printf "%8.4f\n" "$x") # Convert to four decimal places.
	echo "Bash runtime:" $x" seconds" | lolcat
}

# Main driver function.
main() {
	start_time=$(date +"%s%4N") # Start runtime timer.
	looper
	end_time=$(date +"%s%4N") # End runtime timer.
	duration=$(expr $end_time - $start_time) # Calculate runtime duration.
	duration=$(echo "$((duration))/10000" | bc -l ) # Convert milliseconds to seconds.
	printer $duration

}

# Big red activation button.
main
