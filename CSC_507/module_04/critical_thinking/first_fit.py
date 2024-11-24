import memory_block
"""
First Fit Memory Allocation

Memory allocation simulation using the first fit algorithm. The algorithm searches
through a list of memory blocks and allocates the first block that is large enough to 
hold a process. If a process is allocated to a block larger than required, the remaining 
space in the block is split off as a new free memory block.

Usage:
  import first_fit
  allocator = first_fit.FirstFitMemoryAllocator(memory_blocks)
"""
"""
Class to simulate memory allocation using the First-Fit algorithm.
This algorithm allocates the first available block large enough for a given process.
"""
class FirstFitMemoryAllocator:
    """
    Constructor function, initializes the memory allocator with a list of memory blocks.
    
    Args:
        memory_blocks (list): A list of integers representing the initial size of free memory blocks.
    """
    def __init__(self, memory_blocks):
        self.memory_blocks = [] # Initialize an empty list to store memory blocks.
        # Convert the provided list of sizes into MemoryBlock objects.
        for size in memory_blocks:
            block = memory_block.MemoryBlock(size)  # Create a MemoryBlock for each size.
            self.memory_blocks.append(block)  # Add the created block to the memory blocks list.

    """
    Allocate memory to a process using the First-Fit algorithm.
    
    This method will search for the first available block of memory that is large enough
    to accommodate the process. If the process requires less memory than the block provides, 
    the block is split into allocated and free parts.
    
    Args:
        process_id (str): The ID of the process requesting memory.
        process_size (int): The size of memory requested by the process.
    
    Returns:
        bool: True if allocation was successful, False otherwise.
    """
    def allocate(self, process_id, process_size):
        # Iterate over all memory blocks to find a suitable block for the process.
        for block in self.memory_blocks:
            # Check if the block is free and large enough to accommodate the process.
            if not block.is_allocated and block.size >= process_size:
                # Allocate the block to the process.
                block.is_allocated = True
                block.process_id = process_id
                # If the block is larger than needed, split it into an allocated part and a free part.
                if block.size > process_size:
                    remaining_size = block.size - process_size  # Calculate the remaining free space.
                    block.size = process_size  # Resize the allocated block.
                    self.memory_blocks.append(memory_block.MemoryBlock(remaining_size))  # Add the remaining part as a free block.                
                # Print confirmation of successful allocation.
                print(f"Process {process_id} allocated to block with size {block.size}")
                return True
        # If no suitable block was found, print an error message.
        print(f"Process {process_id} could not be allocated. Not enough memory.")
        return False

    """
    Deallocate memory for a specific process by its process ID.
    
    Args:
        process_id (str): The ID of the process to deallocate memory for.
    
    Returns:
        bool: True if deallocation was successful, False otherwise.
    """
    def deallocate(self, process_id):
        # Iterate over all memory blocks to find the block allocated to the given process.
        for block in self.memory_blocks:
            if block.is_allocated and block.process_id == process_id:
                # Free up the block by resetting its allocation status.
                block.is_allocated = False
                block.process_id = None
                # Print confirmation of successful deallocation.
                print(f"Process {process_id} deallocated.")
                return True
        # If the process is not found in any allocated block, print an error message.
        print(f"Process {process_id} not found in allocated blocks.")
        return False

    """
    Display the current state of memory blocks (both allocated and free).
    This method prints out the size, allocation status, and the process ID (if allocated)
    for each memory block.
    """
    def display_memory(self):
        print("\nMemory Blocks:")
        # Iterate over all memory blocks and print their details.
        for block in self.memory_blocks:
            print(block)  # Calls the __str__ method of MemoryBlock, which provides block details.


