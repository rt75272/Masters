import random
import time
import multiprocessing
# ---------------------------------------------------------------------------------------
# Python Number Generator.
#
# Generates a million random integers and saves them to a file named file2.txt. Also,
# calculates and outputs the runtime.
#
# Usage:
#	$ python numbers.py
# ---------------------------------------------------------------------------------------
n = 1000000  # Total number of random integers to generate.
num_cores = multiprocessing.cpu_count()  # Get the number of CPU cores available.
filename = "file2.txt"  # Python output file.

# Formats and displays the final output.
def printer(duration):
	duration = round(duration, 4)  # Convert to have four decimal places.
	print("Python runtime: " + str(duration) + " seconds") 

# Loops through n times and generates random integers, which are saved to file.
def looper(start, count):
	with open(filename, "a") as file:
		for i in range(count):
			x = random.randint(0, n)  # Generate a random integer between 0 and n.
			file.write(f"{x}\n")  # Add the random number with a newline, to the file.

# Break up the looper function into several processes.
def multithread():
	processes = []
	chunk_size = n // num_cores  # Calculate chunk size for each core.
	remainder = n % num_cores  # Calculate remainder to distribute.
	# Loop through all cores and assign each a subset of the overall task.
	for i in range(num_cores):
		count = chunk_size  # Base count for each process.
		if i < remainder:
			count += 1
		process = multiprocessing.Process(target=looper, args=(i, count))  # Declare a new thread/process.
		process.start()  # Start the new process with a mission to produce its assigned number of integers to produce.
		processes.append(process)  # Keep track of the processes.
	# Loop through all separate running processes.
	for process in processes:
		process.join()  # Wait for all processes to finish.

# Main driver function.
def main():
	start_time = time.time()  # Start the runtime timer.
	multithread()
	end_time = time.time()  # End the runtime timer.
	duration = end_time - start_time
	printer(duration)

# The big red activation button.
if __name__ == "__main__":
	main()
	print("numbers.py complete!")
 # type: ignore