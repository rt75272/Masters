from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
# -----------------------------------------------------------------------------
# Merge Sort
#
# Standard merge sort implemenation with a plotting animation to conduct a 
# sorting algorithm comparison.
#
# Usage:
#   import merge
#   merger = merge.MergeSort(array)
# -----------------------------------------------------------------------------
class MergeSort:
    # Constructor function.
    def __init__(self, arr):
        # Array to be sorted.
        self.arr = arr
        # Array of graph frames for animation.
        self.frames = []

    # Sort and return a given array, arr.
    def sort(self):
        self._merge_sort(self.arr)
        self.animate_sorting()
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
        merged = self._merge(sorted_left, sorted_right)
        # Store the frame (the state of the array after merging).
        self.frames.append(merged.copy())
        return merged
    
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
                # add the right array's element to the final merged array.
                merged.append(right[right_index])
                right_index+=1
        # Combine sorted left and right into a single sorted array.
        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        return merged
    
    # Visualize the sorting process.
    def animate_sorting(self):
        # Create the first plot and frame.
        fig, ax = plt.subplots()
        # X-axis values for plotting.
        x = np.arange(len(self.arr))
        # Set to bar graph style.
        bars = ax.bar(x, self.arr, color='b')
        # Set the y-axis and x-axis limits.
        ax.set_ylim(0, max(self.arr)+10)
        ax.set_xlim(-0.5, len(self.arr)-0.5)

        # Update sub-function for animation.
        def update(frame):
            # Pair each bar(s) and frame values together.
            for bar, height in zip(bars, frame):
                # Set max height of bar graph value.
                bar.set_height(height)
            return bars

        # Create the animation.
        anim = FuncAnimation(fig, update, frames=self.frames, repeat=False, blit=True)
        plt.title("Merge Sort")
        # Automatically close the animation figure.
        plt.show(block=False)
        plt.pause(len(self.frames)*0.25)
        plt.close(fig)