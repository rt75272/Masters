import linear_search

# Main driver function.
def main():
    # Build linear search parameters
    arr = [99, 42, 0, 1, 234, -5]
    n = len(arr)
    target = 42

    # Linear search object.
    lin_search = linear_search.LinearSearch()

    # Array index of element matching given target.
    matching_index = lin_search.linear_search(arr, n, target)

    # Display results.
    print(f"Target element is located at index {matching_index}")

if __name__ == "__main__":
    main()