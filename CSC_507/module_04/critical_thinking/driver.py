import first_fit
# --------------------------------------------------------------------------------------------
# First Fit Driver
#
# Uses the first fit class in order to run a memory allocation simulation.
#
# Usage:
#   python driver.py
# --------------------------------------------------------------------------------------------
# Main driver function. 
def main():
	# Initial memory blocks (size in MB).
	memory_blocks = [
		100,
		200,
		300,
		400,
		500
	] 
	allocator = first_fit.FirstFitMemoryAllocator(memory_blocks) # Create memory allocator instance.
	
	# Processes with their required memory sizes.
	processes = [
		(120, "P1"), 
		(250, "P2"), 
		(320, "P3"), 
		(60, "P4"), 
		(450, "P5")
	] 
	
	# Allocate processes.
	for process_size, process_id in processes:
		if not allocator.allocate(process_size, process_id):
			print(f"Process {process_id} could not be allocated due to insufficient memory.")
	
	allocator.display_memory_status() # Display memory status.
	
	# Deallocate some processes and display the memory status again.
	allocator.deallocate("P2")
	allocator.deallocate("P3")
	allocator.display_memory_status()
	
	# Try allocating another process.
	allocator.allocate(100, "P6")
	allocator.display_memory_status()

# The big red activation button.
if __name__ == "__main__":
	main()