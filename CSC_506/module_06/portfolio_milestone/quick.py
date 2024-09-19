# -----------------------------------------------------------------------------
# Quick Sort
#
# Standard quick sort implemenation.
#
# Usage:
#   import quick
#   quick = quick.QuickSort(array)
# -----------------------------------------------------------------------------
class QuickSort:
    # Constructor function.
    def __init__(self, arr):
        self.arr = arr # Array to be sorted.

    # Sort and update a given array.
    def sort(self, frames):
        self._quicksort(0, len(self.arr)-1, frames)
        frames.append(self.arr.copy()) # Final sorted array.

    # Private quick sort implementation.
    def _quicksort(self, low, high, frames):
        # Base case. Check if the segment has one or no elements. Its already sorted.
        if low < high:            
            pivot_index = self._partition(low, high) # Partition the array and get the pivot point.            
            frames.append(self.arr.copy()) # Save a frame of the current state of the array.            
            self._quicksort(low, pivot_index-1, frames) # Recursive quicksort call with thte left segment[low : pivot point].            
            self._quicksort(pivot_index+1, high, frames) # Recursive quicksort call with the right segment[pivot point : high].

    # Private partitioning function to support _quicksort.
    def _partition(self, low, high):
        pivot = self.arr[high] # Establish the pivot point as the last element of the array.
        i = low - 1 # Set the partition index(i) to the lowest element.
        # Loop through all elements in a given array segment.
        for j in range(low, high):
            # Check if the current element is less than or equal to the pivot.
            if self.arr[j] <= pivot:
                # Move the partition index to the right.
                i+=1
                tmp = self.arr[i] # Temp holder for our lowest array element.
                self.arr[i] = self.arr[j] # Move the right element to the left.
                self.arr[j] = tmp # Assign the right index with the previous left value, stored in the temp variable.
        # Place the pivot point in its correct position by swapping it with the element at the partition index.
        temp_pivot = self.arr[i+1] # Temp holder for our first arrray element.
        self.arr[i+1] = self.arr[high] # Swap the pivot element with the element at index i. 
        self.arr[high] = temp_pivot # Assign the pivot element in position i.
        return i+1 # Return the index of the pivot point.