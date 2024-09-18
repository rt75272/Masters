#------------------------------------------------------------------------------
# Node 
#
# Standard python node class.
#
# Usage:
#   import node
#   new_node = node.Node()
#------------------------------------------------------------------------------
class Node:
    # Constructor function.
    def __init__(self, key, value):
        self.key = key # Key value to access node.
        self.value = value # Content contained by the node.
        self.next = None # Pointer to the next node.
