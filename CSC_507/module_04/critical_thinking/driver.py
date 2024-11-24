import first_fit
# --------------------------------------------------------------------------------------------
# First Fit Driver
#
# Uses the first fit class in order to run a memory allocation simulation.
#
# Usage:
#   python driver.py
# --------------------------------------------------------------------------------------------
"""
Main driver function. 

Runs the first fit memory allocation simulation.
"""
def main():
	# Initial memory blocks sizes.
	memory_sizes = [101, 507, 222, 303, 666]
	# Create a memory allocator with the initial memory sizes.
	allocator = first_fit.FirstFitMemoryAllocator(memory_sizes)
	# Display memory before allocation.
	allocator.display_memory()
	# Allocate memory for processes.
	allocator.allocate("P1", 120)
	allocator.allocate("P2", 42)
	allocator.allocate("P3", 1)
	allocator.allocate("P4", 3274)
	# Display memory after allocation.
	allocator.display_memory()
	# Deallocate a process.
	allocator.deallocate("P2")
	# Display memory after deallocation.
	allocator.display_memory()

# The big red activation button.
if __name__ == "__main__":
	main()

