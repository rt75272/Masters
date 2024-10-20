# --------------------------------------------------------------------------------------------
# First Fit Memory Allocation
#
# Memory allocation simulation using the first fit algorithm. Generates blocks and processes 
# of specific sizes. The first fit algorithm will search sequentially until it finds a memory 
# block greater than or equal to the size of the process. At which point the process will be 
# placed in the block of memory in order to execute. 
#
# Usage:
#   import first_fit
#   allocator = first_fit.FirstFitMemoryAllocator(memory_blocks)
# --------------------------------------------------------------------------------------------
class FirstFitMemoryAllocator:
    # Constructor function.
    def __init__(self, memory_blocks):
        self.memory_blocks = memory_blocks # Initialize memory blocks (list of block sizes)
        self.allocations = [None] * len(memory_blocks) # Initialize a list to track if memory block is allocated (None if unallocated)

	# --------------------------------------------------------------------------------------------
	# Try to allocate a memory block for the process using First-Fit.
    #
	# :param process_size: Size of the process to allocate
	# :param process_id: Identifier of the process
	# :return: True if allocated successfully, False if no suitable block is found
	# --------------------------------------------------------------------------------------------
    def allocate(self, process_size, process_id):
        for i in range(len(self.memory_blocks)):
            if self.memory_blocks[i] >= process_size and self.allocations[i] is None:
                # Allocate the process to this block
                self.allocations[i] = process_id
                print(f"Process {process_id} allocated to Block {i} (Size: {self.memory_blocks[i]})")
                return True
        return False

	# --------------------------------------------------------------------------------------------
	# Deallocate the memory block occupied by a process.
	#
    # :param process_id: Identifier of the process to deallocate
	# :return: None
	# --------------------------------------------------------------------------------------------
    def deallocate(self, process_id):
        for i in range(len(self.allocations)):
            if self.allocations[i] == process_id:
                print(f"Process {process_id} deallocated from Block {i}")
                self.allocations[i] = None
                return

	# --------------------------------------------------------------------------------------------
	# Display the current status of memory blocks and allocations.
    #
    # :return: None
	# --------------------------------------------------------------------------------------------
    def display_memory_status(self):
        print("\nMemory Blocks Status:")
        for i in range(len(self.memory_blocks)):
            if self.allocations[i]:
                status = f"Block {i}: {self.memory_blocks[i]} (Allocated to Process {self.allocations[i]})"
            else:
                status = f"Block {i}: {self.memory_blocks[i]} (Free)"
            print(status)