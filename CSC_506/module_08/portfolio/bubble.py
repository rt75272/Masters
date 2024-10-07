from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
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
    # Constructor function.
    def __init__(self, arr):
        self.arr = arr # Array to be sorted.

    # Sort and return a given array, arr.
    def sort(self, frames):
        self._bubble_sort(self.arr, frames)

    # Private bubble sort implementation.
    # Plots and returns the bubbly sorted array.
    def _bubble_sort(self, arr, frames):
        n = len(arr) # Grab the number of elements in the array.
        frames.append(arr.copy())  # Store initial frame.
        # Loop through all elements in the array.
        for i in range(n):
            # Inner loop in order to compare elements next to each other.
            for j in range(n-i-1):
                # Check if the element to the left is larger than the element to the right.
                if(arr[j] > arr[j+1]):
                    tmp = arr[j] # Temp assignment and holder for arr[j]'s value.
                    arr[j] = arr[j+1] # Move the smaller element on the right, to the left. 
                    arr[j+1] = tmp # Using the temp holder, move the larger value on the left, to the right.
                    frames.append(arr.copy()) # Store a copy of the array for the animation.