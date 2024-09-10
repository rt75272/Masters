import bubble
import numpy
import time

# Main driver function.
def main():
    # Array of random integers to be sorted.
    x = numpy.random.randint(100, size=(10000))
    
    # Establish bubble sort object.
    bubbler = bubble.BubbleSort()

    # Start bubble sort timer.
    bubble_start = time.time()

    # Run bubble sort algorithm.
    bubbler.bubble_sort(x)
    
    # End bubble sort timer.
    bubble_end = time.time()

    # Calculate bubble sort runtime. 
    bubble_time = bubble_end - bubble_start
    print(bubble_time)

# Big red button.
if __name__ == "__main__":
    main()
