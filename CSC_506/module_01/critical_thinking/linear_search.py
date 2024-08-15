# Linearly searches an array for a given target.
class LinearSearch:
    def __init__(self):
        pass

    def linear_search(self, arr, n, target):
        for i in range(n):
            if(arr[i] == target):
                return i
        return -1