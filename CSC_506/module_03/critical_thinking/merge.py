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
    # Constructor.
    def __init__(self, arr):
        self.arr = arr

    # Sort and return a given array, arr.
    def sort(self):
        self.arr = self._merge_sort(self.arr)
        return self.arr

    # Private merge sort implementation.
    def _merge_sort(self, arr):
        # Check for an empty array.
        if len(arr) <= 1:
            return arr
        
        # Get the middle point of the array.
        mid = len(arr) // 2

        # Assign left half, 0 - mid.
        left_half = arr[:mid]

        # Assign right half, mid - end.
        right_half = arr[mid:]

        # Sort each half sepearately.
        sorted_left = self._merge_sort(left_half)
        sorted_right = self._merge_sort(right_half)

        # Merge and return the sorted halves.
        return self._merge(sorted_left, sorted_right)
    
    # Private merge implementation.
    # Merges two given arrays into a single merged together array.
    # Returns the merged array.
    def _merge(self, left, right):
        # To be the merged array.
        merged = []

        # Initial left and right side indexes.
        left_index = 0
        right_index = 0

        # Run while indexes are less then thier respective lengths.
        while(left_index < len(left) and right_index < len(right)):
            # Check if the left array's element is less than the right array's element.
            if(left[left_index] <= right[right_index]):
                # If so, add the left array's element to the final merged array.
                merged.append(left[left_index])
                left_index+=1
            else:
                # If the left array's element is greater than the right array's element,
                #   add the right array's element to the final merged array.
                merged.append(right[right_index])
                right_index+=1
        
        # Combine sorted left and right into a single sorted array.
        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        return merged