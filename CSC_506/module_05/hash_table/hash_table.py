from node import Node
#------------------------------------------------------------------------------
# Hash Table
#
# Standard hash table implementation.
#
# Usage:
#   import hash_table
#   new_hash_table = hash_table.HashTable()
#------------------------------------------------------------------------------
class HashTable:
    # Constructor function.
    def __init__(self, size=42):
        self.size = size # Initialize the hash table with a given size.
        self.table = [None]*size # Initialize with None for linked list head pointers.

    # Compute hash value for given key.
    def _hash(self, key):
        hashed_key = hash(key) % self.size
        return hashed_key

    # Insert a key-value pair into the hash table.
    def insert(self, key, value):
        index = self._hash(key) # Get the index, which is its own hash key.
        new_node = Node(key, value) # Initialize a new node.
        # If the bucket is empty, add the new node.
        if self.table[index] is None:
            self.table[index] = new_node # Add the new node.
        # If the bucket is not empty.
        else:
            current = self.table[index] # Assign current to be at the provided index.
            # Check all nodes for an existing hash key, or find the end.
            while current:
                # Check for key hash values.
                if current.key == key:
                    current.value = value # Key exists, update the value.
                    return # End.
                # Check for no next-node.
                if current.next is None:
                    break # End.
                current = current.next  # Assign the next node to current.
            current.next = new_node # Insert the new node at the end of the list, using the next pointer.

    # Retrieve the value associated with the given key.
    def get(self, key):
        index = self._hash(key) # Index of the node.
        current = self.table[index] # Grab the current node.
        # Loop while there is a current node.
        while current:
            # Check if the current's node matches the target key.
            if current.key == key:
                return current.value # Return the matching node's value.
            current = current.next # Assign the next node to current.
        return None # Key not found, return none.

    # Delete the key-value pair associated with the given key.
    def delete(self, key):
        index = self._hash(key) # Index of the node.
        current = self.table[index] # Assign the node to current.
        prev = None # Set the previous-node-pointer to none.
        # Loop while there is a next node.
        while current:
            # Check if the currrent node's key matches the target key.
            if current.key == key:
                # Check for no previous nodes.
                if prev is None:
                    self.table[index] = current.next # Removing the head node. Removes pointer to self node.
                else:
                    prev.next = current.next # Removing a node that is not the head. Removes pointer to self node.
                return True # Keep looping.
            prev = current # Assign current to previous.  
            current = current.next # Assign the next node to current.
        return False # Key not found

    # Printer function.
    def __repr__(self):
        ret_val = "" # To be our final output string.
        items = [] # List of items to be printed.
        # Enumerate through the hash table.
        for i, head in enumerate(self.table):
            current = head # Assign the head node to current.
            # Keep looping while nodes exists.
            while current:
                items.append(f'{current.key}: {current.value}') # Add key-pair value to our array of items to be printed.
                current = current.next # Assign the next node to current.
        # Build and return the final output string.
        ret_val = '{' + ','.join(items) + '}'
        return ret_val
