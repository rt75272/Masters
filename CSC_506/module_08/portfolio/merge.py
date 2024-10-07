# -----------------------------------------------------------------------------
# Merge Sort
#
# Standard merge sort implemenation.
#
# Usage:
#   import merge
#   merger = merge.MergeSort(array)
# -----------------------------------------------------------------------------
class MergeSort:
    # Constructor function.
    def __init__(self, arr):
        self.arr = arr # Array to be sorted.

    # Sort and return a given array, arr.
    def sort(self, frames):
        self._merge_sort(self.arr, frames)
        frames.append(self.arr.copy()) # Get copy of final sorted array.

    # Private merge sort implementation.
    def _merge_sort(self, arr, frames):
        # Check for an empty array.
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2 # Get the middle point of the array.
        left_half = self._merge_sort(arr[:mid], frames) # Assign left half, 0 - mid.
        right_half = self._merge_sort(arr[mid:], frames) # Assign right half, mid - end.
        merged = self._merge(left_half, right_half) # Combine the two halves.
        frames.append(merged.copy()) # Store the frame (the state of the array after merging).
        # Update and return the original array.
        for i in range(len(merged)):
            arr[i] = merged[i]
        return arr
    
    # Private merge implementation.
    # Merges two given arrays into a single merged together array.
    # Returns the merged array.
    def _merge(self, left, right):
        merged = [] # To be the merged array.
        # Initial left and right side indexes.
        left_index = 0
        right_index = 0
        # Run while indexes are less then thier respective lengths.
        while(left_index < len(left) and right_index < len(right)):
            # Check if the left array's element is less than the right array's element.
            if(left[left_index] <= right[right_index]):
                merged.append(left[left_index]) # Add the left array's element to the final merged array.
                left_index+=1
            # If the left array's element is greater than the right array's element.
            else:
                merged.append(right[right_index]) # Add the right array's element to the final merged array.
                right_index+=1
        # Combine sorted left and right into a single sorted array.
        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        return merged