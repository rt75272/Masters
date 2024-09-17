import bubble
import merge
import numpy
import quick
import time
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

    # Create copies of the random array for fair comparison.
    arr_for_bubble = random_arr.copy()
    arr_for_merge = random_arr.copy()
    arr_for_quicky = random_arr.copy()
    
    # Establish bubble sort and merge sort objects.
    bubbler = bubble.BubbleSort(arr_for_bubble)
    merger = merge.MergeSort(arr_for_merge)
    quicky = quick.QuickSort(arr_for_quicky)
    
    # Run and time sorting algorithms.
    bubble_time = calculate_runtime(bubbler)
    merge_time = calculate_runtime(merger)
    quick_time = calculate_runtime(quicky)

    # Display sorting algorithms' runtimes.
    print(f"Bubble sort runtime:\t {bubble_time:.4f} seconds")
    print(f"Merge sort runtime:\t {merge_time:.4f} seconds")
    print(f"Quick sort runtime:\t {quick_time:.4f} seconds")

# Run the given algorithm in order to measure its runtime.
def calculate_runtime(algorithm):
    start_time = time.time()
    algorithm.sort()
    elapsed_time = time.time() - start_time
    return elapsed_time

# Generate and returns an array of random integers.
def get_random_arr():
    n = 20
    print(f"Random array size(n): {n}")
    arr = numpy.random.randint(100, size=(n))
    return arr

# Big red button.
if __name__ == "__main__":
    main()
