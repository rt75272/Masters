# -----------------------------------------------------------------------------
# Insertion Sort
#
# Standard insertion sort implemenation.
#
# Usage:
#   import insertion
#   insertion = insertion.InsertionSort(array)
# -----------------------------------------------------------------------------
class InsertionSort:
    # Constructor function.
    def __init__(self, arr):
        self.arr = arr # Array to be sorted.

    # Sort and update a given array.
    def sort(self, frames):
        self._insertion_sort(self.arr, frames)

    # Private insertion sort implementation.
    def _insertion_sort(self, arr, frames):
        frames.append(arr.copy()) # Add given array to frames array.
        # Loop through the elements.
        for i in range(1, len(arr)):
            key = arr[i] # Assign current array element as key.
            j = i-1 # Assign value of i-1 to variable j.
            # Loop while greater than zero and key is less than the array element. 
            while j >= 0 and key < arr[j]:
                arr[j+1] = arr[j] # Move elment right.
                j-=1 # Decrement.
            arr[j+1] = key 
            frames.append(arr.copy()) # Capture frame.
