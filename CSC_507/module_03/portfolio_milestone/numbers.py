import random
import time
import multiprocessing
from termcolor import colored, cprint
# ---------------------------------------------------------------------------------------
# Python Number Generator.
#
# Generates a million random integers and saves them to a file named file2.txt. Also,
# calculates and outputs the runtime.
#
# Usage:
#	$ python numbers.py
# ---------------------------------------------------------------------------------------
# Initial setup.
n = 1000000 # Upper limit.
num_cores = multiprocessing.cpu_count() # Get the number of cpu cores available.
n = n // num_cores # Divide the total n into the number of cores. Each core will run a chunk of n.
filename = "file2.txt" # Python output file.

# Formats and displays the final output.
def printer(time):
	time = round(time, 4) # Convert to have four decimal places.
	print(colored(
		"Python runtime: " 
		+ str(time) + " seconds",
		"cyan", attrs=["bold"]) 
	)

# Loops through n times and generates n random integers, which are saved to file.
def looper():
	file = open(filename, "a") # Creates(if needed) and opens the file to be appended to.
	# Loop from zero to one million.
	for i in range(n):
		x = random.randint(0, n) # Generate a random integer between 0 and n.
		x = str(x) + "\n" # Convert the variable x to a string with a newline. 
		file.write(x) # Append the value of x to our file.
	file.close() # Close the output file.

# Break up the looper function into several threads.
def multithread():
	# Loop through the number of cpu cores and divide looper up into each core.
	for i in range(num_cores):
		process = multiprocessing.Process(target=looper)
		process.start()

# Main driver function.
def main():
	start_time = time.time() # Start the runtime timer.
	multithread()
	end_time = time.time() # End the runtime timer.
	duration = end_time - start_time
	printer(duration)

# The big red activation button.
if __name__ == "__main__":
	main()

# type: ignore ----> Suppress vvscode warnings.

