# Bubble sort object.
class BubbleSort:
    # Constructor.
    def __init__(self):
        pass

    # Standard bubble sort.
    def bubble_sort(self, arr):
        n = len(arr)

        for i in range(n):
            print(i)

            for j in range(n-i-1):
                if(arr[j] > arr[j+1]):
                    tmp = arr[j]
                    arr[j] = arr[j+1]
                    arr[j+1] = tmp