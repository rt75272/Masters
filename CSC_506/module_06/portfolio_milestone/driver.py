import time
import merge
import quick
import bubble
import insertion
import selection
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
plt.style.use('dark_background') # Enable dark mode for the sorting animation.

# Global dictionary of lists to store frames for animations.
frames = {
    "bubble": [],
    "merge": [],
    "quick": [],
    "insertion": [],
    "selection": []
}

# Function to run sorting algorithms.
def run_sorting(algorithm, name):
    algorithm.sort(frames[name])

# Main driver function.
def main():
    random_arr = get_random_arr() # Array of random integers to be sorted.
    # Create copies of the random array for fair comparison.
    arr_for_bubble = random_arr.copy()
    arr_for_merge = random_arr.copy()
    arr_for_quicky = random_arr.copy()
    arr_for_insertion = random_arr.copy()
    arr_for_selection = random_arr.copy()
    # Establish bubble sort and merge sort objects.
    bubbler = bubble.BubbleSort(arr_for_bubble)
    merger = merge.MergeSort(arr_for_merge)
    quicky = quick.QuickSort(arr_for_quicky)
    inserter = insertion.InsertionSort(arr_for_insertion)
    selector = selection.SelectionSort(arr_for_selection) 
    # Setup threads in order to run all algorithms at the same time.
    bubble_thread = threading.Thread(target=run_sorting, args=(bubbler, "bubble"))
    merge_thread = threading.Thread(target=run_sorting, args=(merger, "merge"))
    quick_thread = threading.Thread(target=run_sorting, args=(quicky, "quick"))
    insertion_thread = threading.Thread(target=run_sorting, args=(inserter, "insertion"))
    selection_thread = threading.Thread(target=run_sorting, args=(selector, "selection")) 
    # Start the threads.
    bubble_thread.start()
    merge_thread.start()
    quick_thread.start()
    insertion_thread.start()
    selection_thread.start()
    # Combine all threads.
    bubble_thread.join()
    merge_thread.join()
    quick_thread.join()
    insertion_thread.join()
    selection_thread.join()
    animate_all_sorts(bubbler, merger, quicky, inserter, selector) # Run the entire sorting animation.
    # Display sorting algorithms.
    print("\n\tFINAL SORTED RESULTS:")
    print(f"\n\tBubble Sort:\t {bubbler.arr}")
    print(f"\n\tMerge Sort:\t {merger.arr}")
    print(f"\n\tQuick Sort:\t {quicky.arr}")
    print(f"\n\tInsertion Sort:\t {inserter.arr}")  
    print(f"\n\tSelection Sort:\t {selector.arr}")

# Runs an animation displaying all sorting algorithms at the same time.
def animate_all_sorts(bubbler, merger, quicky, inserter, selector):
    fig, ax = plt.subplots(5, 1, figsize=(10, 25)) # Plotting setup.
    # Set titles for each respective algorithm.
    ax[0].set_title("Bubble Sort", color='white')
    ax[1].set_title("Merge Sort", color='white')
    ax[2].set_title("Quick Sort", color='white')
    ax[3].set_title("Insertion Sort", color='white')
    ax[4].set_title("Selection Sort", color='white')
    # Setup graph bars for each algorithm.
    bars_bubble = ax[0].bar(np.arange(len(bubbler.arr)), bubbler.arr, color='g')
    bars_merge = ax[1].bar(np.arange(len(merger.arr)), merger.arr, color='b')
    bars_quick = ax[2].bar(np.arange(len(quicky.arr)), quicky.arr, color='r')
    bars_insertion = ax[3].bar(np.arange(len(inserter.arr)), inserter.arr, color='y')  
    bars_selection = ax[4].bar(np.arange(len(selector.arr)), selector.arr, color='m')
    # Adjust y-limits for the graph.
    for a in ax:
        a.set_ylim(0, max(max(bubbler.arr), max(merger.arr), max(quicky.arr), max(inserter.arr), max(selector.arr)) + 5)
    # Update each graph section, for each algorithm.
    def update_all(frame):
        if frame < len(frames["bubble"]):
            for bar, height in zip(bars_bubble, frames["bubble"][frame]):
                bar.set_height(height)
        if frame < len(frames["merge"]):
            for bar, height in zip(bars_merge, frames["merge"][frame]):
                bar.set_height(height)
        if frame < len(frames["quick"]):
            for bar, height in zip(bars_quick, frames["quick"][frame]):
                bar.set_height(height)
        if frame < len(frames["insertion"]):
            for bar, height in zip(bars_insertion, frames["insertion"][frame]):
                bar.set_height(height)
        if frame < len(frames["selection"]):
            for bar, height in zip(bars_selection, frames["selection"][frame]):
                bar.set_height(height)
        # Return the current graph bars 
        return bars_bubble + bars_merge + bars_quick + bars_insertion + bars_selection
    # Sorting Animaiton.
    total_frames = max(len(frames["bubble"]), len(frames["merge"]), len(frames["quick"]), len(frames["insertion"]), len(frames["selection"]))  # Update total frames.
    anim = FuncAnimation(fig, update_all, frames=total_frames, repeat=False)
    # Automatically close the animation figure.
    plt.show(block=False)
    plt.pause(len(frames["bubble"])*0.25) # Pause time based on bubble sort cause its the slowest.
    plt.close(fig) # Shut it down.

# Run the given algorithm in order to measure its runtime.
def calculate_runtime(algorithm):
    start_time = time.time()
    algorithm.sort()
    elapsed_time = time.time() - start_time
    return elapsed_time

# Generate and returns an array of random integers.
def get_random_arr():
    n = 30 # Number of elements to be randomly generated and sorted.
    print(f"Random array size(n): {n}")
    arr = np.random.randint(100, size=(n)) # Array of randomly generated integers.
    return arr

# Big red button.
if __name__ == "__main__":
    main()