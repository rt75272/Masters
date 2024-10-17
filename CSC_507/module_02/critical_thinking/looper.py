import time
import multiprocessing
from termcolor import cprint # type: ignore
# ---------------------------------------------------------------------------- 
# Neverending Looper.
#
# Runs a neverending loop in order to practice using process managers.
#
# Usage:
#	$ python looper.py
# ----------------------------------------------------------------------------
# Constant variables - just cause, better readability?
_ZERO = 0
_ONE = 1
_TWO = 2

# Determines and returns the color to be used.
def get_color(n):
	options = ["cyan", "light_cyan", "light_blue"] # Array of color options.
	selection = "red" # Default color for when something breaks.
	# Determine the color selection.
	idx = n-_ONE
	if(idx == _ZERO):
		selection = options[_ZERO]
	elif(idx == _ONE):
		selection = options[_ONE]
	elif(idx == _TWO):
		selection = options[_TWO]
	else:
		print("Something broke")
	return selection # Return the selected color.

# Neverending looper function.
def looper():
	counter = _ZERO # Standard loop counter.
	infinite = True
	while infinite:	# Loops forever. 
		num_hola = (counter % 3 + _ONE) # Calculate the number of greetings.
		color_option = get_color(num_hola) # Grab the color option.
		cprint("Hello " * num_hola * _TWO, color_option) # Colored output.
		counter+=_ONE
		# time.sleep(0.5)

# Spreads the looper function to every available cpu core.
def multithread():
    jobs = [] # Array of thread jobs.
	# Loop through the number of cpu cores available.
    for i in range(multiprocessing.cpu_count()): 
        process = multiprocessing.Process(target=looper) # Create a process object with looper.
        jobs.append(process) # Add the process to our array of processes/jobs.
        process.start() # Start the thread.

# # Big red activation button.
if __name__ == "__main__":
	multithread()