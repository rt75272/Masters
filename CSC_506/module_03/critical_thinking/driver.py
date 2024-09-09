import bubble
import numpy

def main():
    x = numpy.random.randint(100, size=(1000))

    # for i in range(999):
    #     x.append(random.randint(0,999))

    bubbler = bubble.BubbleSort()

    bubbler.bubble_sort(x)


if __name__ == "__main__":
    main()
