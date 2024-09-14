import bubble
import numpy
import time
import merge
# -----------------------------------------------------------------------------
# Main
# 
# Main driver for the bubble sort and merge sort implemenations. Makes use of
# an array of random integers to be sorted by both bubble sort and merge sort. 
# Then compares the runtimes of both sorting algorithms against each other.
#
# Usage:
#   $ python dirver.py
# -----------------------------------------------------------------------------
# Main driver function.
def main():
    # Array of random integers to be sorted.
    random_arr = get_random_arr()
    
    # Establish bubble sort and merge sort objects.
    bubbler = bubble.BubbleSort(random_arr)
    merger = merge.MergeSort(random_arr)
    
    # Run and time sorting algorithms.
    bubble_time = calculate_runtime(bubbler)
    merge_time = calculate_runtime(merger)

    print(f"Bubble sort runtime: {bubble_time}")
    print(f"Merge sort runtime: {merge_time}")
  
# Run the given algorithm in order to measure its runtime.
def calculate_runtime(algorithm):
    start_time = time.time()
    algorithm.sort()
    elapsed_time = time.time() - start_time
    return elapsed_time

# Generate and returns an array of random integers.
def get_random_arr():
    n = 10000
    print(f"Random array size(n): {n}")
    arr = numpy.random.randint(100, size=(n))
    return arr

# Big red button.
if __name__ == "__main__":
    main()
