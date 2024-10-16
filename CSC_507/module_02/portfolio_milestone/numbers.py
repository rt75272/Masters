import random
# ---------------------------------------------------------------------------------------
# Python Number Generator.
#
# Generates a thousand random integers and saves them to a file named file2.txt.
#
# Usage:
#	$ python numbers.py
# ---------------------------------------------------------------------------------------
# Initial setup.
n = 1000 # Upper limit.
filename = "file2.txt" # Python output file.

# Main driver function.
def main():
	file = open(filename, "a") # Creates(if needed) and opens the file to be appended to.
	# Loop from zero to one thousand.
	for i in range(n):
		x = random.randint(0, n) # Generate a random integer between 0 and 1000.
		x = str(x) + "\n" # Convert the variable x to a string with a newline. 
		file.write(x) # Append the value of x to our file.
	file.close() # Close the output file.

# The big red activation button.
if __name__ == "__main__":
	main()

# type: ignore ----> Suppress vvscode warnings.

