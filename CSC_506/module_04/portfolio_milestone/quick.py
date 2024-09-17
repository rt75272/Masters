import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
# -----------------------------------------------------------------------------
# Quick Sort
#
# Standard quick sort implemenation with a plotting animation to conduct a 
# sorting algorithm comparison.
#
# Usage:
#   import quick
#   quick = quick.QuickSort(array)
# -----------------------------------------------------------------------------
class QuickSort:
    # Constructor function.
    def __init__(self, arr):
        # Array to be sorted.
        self.arr = arr
        # Array of graph frames for animation.
        self.frames = []

    # Sort and return a given array.
    def sort(self):
        self._quicksort(0, len(self.arr)-1)
        self.animate_sorting()
        return self.arr

    # Private quick sort implementation.
    def _quicksort(self, low, high):
        # Base case.
        # Check if the segment has one or no elements. Its already sorted.
        if low < high:
            # Partition the array and get the pivot point.
            pivot_index = self._partition(low, high)
            # Save a frame of the current state of the array.
            self.frames.append(self.arr.copy())
            # Recursive quicksort call with thte left segment[low : pivot point].
            self._quicksort(low, pivot_index-1)
            # Recursive quicksort call with the right segment[pivot point : high].
            self._quicksort(pivot_index+1, high)

    # Private partitioning function to support _quicksort.
    def _partition(self, low, high):
        # Establish the pivot point as the last element of the array.
        pivot = self.arr[high]
        # Set the partition index(i) to the lowest element.
        i = low
        # Loop through all elements in a given array segment.
        for j in range(low, high):
            # Check if the current element is less than or equal to the pivot.
            if self.arr[j] <= pivot:
                # If its less than or equal to pivot, 
                # Temp holder for our lowest array element.
                tmp = self.arr[i]
                # Move the right element to the left.
                self.arr[i] = self.arr[j]
                # Assign the right index with the previous left value, stored in the temp variable.
                self.arr[j] = tmp
                # Move the partition index to the right.
                i+=1
        # Place the pivot point in its correct position by swapping it with the element at the partition index.
        # Temp holder for our first arrray element.
        temp_pivot = self.arr[i]
        # Swap the pivot element with the element at index i. 
        self.arr[i] = self.arr[high]
        # Assign the pivot element in position i.
        self.arr[high] = temp_pivot

        # Return the index of the pivot point.
        return i

    # Visualize the sorting process.
    def animate_sorting(self):
        # Create the first plot and frame.
        fig, ax = plt.subplots()
        # X-axis values for plotting.
        x = np.arange(len(self.arr))
        # Set to bar graph style.
        bars = ax.bar(x, self.arr, color='r')
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
        plt.title("Quick Sort")
        # Automatically close the animation figure.
        plt.show(block=False)
        plt.pause(len(self.frames)*0.25)
        plt.close(fig)