class merge:
    def __init__(self):
        pass

    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2

        # Assign left half, 0-mid.
        left = arr[:mid]

        # Assign right half, mid-end.
        right = arr[mid:]

        sorted_left = merge_sort(left)
        sorted_right = merge_sort(right)

        return merge(sorted_left, sorted_right)
    
    def merge(left, right):
        result = []
        i = 0
        j = 0

        while(i < len(left) and j < len(right)):
            if(left[i] < right[j]):
                result.append(left[i])
                i+=1
            else:
                result.append(right[j])
                j+=1
        
        # Combine sorted left and right into a single sorted array.
        result.extend(left[i:])
        result.extend(right[j:])

        return result