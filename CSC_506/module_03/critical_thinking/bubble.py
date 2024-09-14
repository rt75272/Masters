# -----------------------------------------------------------------------------
# Bubble Sort
#
# Standard bubble sort implementation.
# 
# Usage:
#   import bubble
#   bubbler = bubble.BubbleSort()
# -----------------------------------------------------------------------------
class BubbleSort:
    # Constructor.
    def __init__(self, arr):
        self.arr = arr

    # Sort and return a given array, arr.
    def sort(self):
        self.arr = self._bubble_sort(self.arr)
        return self.arr 

    # Private bubble sort implementation.
    def _bubble_sort(self, arr):
        # Grab the number of elements in the array.
        n = len(arr)

        # Loop through all elements in the array.
        for i in range(n):
            # Inner loop in order to compare elements next to each other.
            for j in range(n-i-1):
                # Check if the element to the left is larger than the element to the right.
                if(arr[j] > arr[j+1]):
                    # Temp assignment and holder for arr[j]'s value.
                    tmp = arr[j]
                    # Move the smaller element on the right, to the left. 
                    arr[j] = arr[j+1]
                    # Using the temp holder, move the larger value on the left, to the right.
                    arr[j+1] = tmp
        return arr