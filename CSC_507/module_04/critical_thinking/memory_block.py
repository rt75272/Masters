# --------------------------------------------------------------------------------------------
# Memory Block
#
# Data structure representing a block of memory. 
#
# Usage:
#   import memory_block
#   memory = memory_block.MemoryBlock(size)
# --------------------------------------------------------------------------------------------
class MemoryBlock:
    """
    Constructor function. Initializes the memory block data structure.
    """
    def __init__(self, size, is_allocated=False, process_id=None):
        self.size = size
        self.is_allocated = is_allocated
        self.process_id = process_id

    """
    Printer function. Builds up and returns a string containing the memory block's information.
    """
    def __str__(self):
        # Prepare the process_id string based on whether the process is allocated.
        process_info = self.process_id if self.process_id else 'None'
        # Return a string representation of the block.
        return f"Block(Size: {self.size}, Allocated: {self.is_allocated}, Process: {process_info})"
    
    
