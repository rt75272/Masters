# -----------------------------------------------------------------------------
# Selection Sort
#
# Standard selection sort implemenation.
#
# Usage:
#   import selection
#   selection = selection.SelectionSort(array)
# -----------------------------------------------------------------------------
class SelectionSort:
    # Constructor function.
    def __init__(self, arr):
        self.arr = arr # Array to be sorted.

    # Sort and update a given array.
    def sort(self, frames):
        self._selection_sort(self.arr, frames)

    # Private selection sort implementation.
    def _selection_sort(self, arr, frames):
        n = len(arr) # Number of elements.
        # Loop through all elements.
        for i in range(n):
            min_idx = i # Assign variable i as min_idx.
            # Inner loop to compare against all other elements.
            for j in range(i + 1, n):
                # Check if current index element is less than the min.
                if arr[j] < arr[min_idx]:
                    min_idx = j # If so, assign variable j as min_idx.
            # Swap elements.
            temp = arr[i] # Temp holder of arr[i]'s value.
            arr[i] = arr[min_idx]
            arr[min_idx] = temp
            frames.append(arr.copy()) # Capture frame after each swap.
