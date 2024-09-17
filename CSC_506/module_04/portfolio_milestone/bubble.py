from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
# -----------------------------------------------------------------------------
# Bubble Sort
#
# Standard bubble sort implementation with a plotting animation to conduct a 
# sorting algorithm comparison.
# 
# Usage:
#   import bubble
#   bubbler = bubble.BubbleSort()
# -----------------------------------------------------------------------------
class BubbleSort:
    # Constructor function.
    def __init__(self, arr):
        # Array to be sorted.
        self.arr = arr
        # Array of graph frames for animation.
        self.frames = []

    # Sort and return a given array, arr.
    def sort(self):
        self._bubble_sort(self.arr)
        self.animate_sorting()
        return self.arr 

    # Private bubble sort implementation.
    # Plots and returns the bubbly sorted array.
    def _bubble_sort(self, arr):
        # Grab the number of elements in the array.
        n = len(arr)
        # Store initial frame.
        self.frames.append(arr.copy())
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
                    # Store a copy of the array for the animation.
                    self.frames.append(arr.copy())
        
    # Animate sorting process.
    def animate_sorting(self):
        # Create first plot and frame.
        fig, ax = plt.subplots()
        # X-axis values for plotting.
        x = np.arange(len(self.arr))
        # Set to bar graph style.
        bars = ax.bar(x, self.arr, color='g')
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
        anim = FuncAnimation(fig, update, frames=self.frames, repeat=False)
        plt.title("Bubble Sort")
        # Automatically close the animation figure.
        plt.show(block=False)
        plt.pause(len(self.frames)*0.25)
        plt.close(fig)