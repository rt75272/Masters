import cv2  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------
# Handwritten Text Recognition with Morphological Operations.
#
# Applies morphological operations (dilation, erosion, opening, closing) on a binary image. The 
# results are displayed in a side-by-side comparison grid.
#
# Usage:
#   $ python morphology.py
# ---------------------------------------------------------------------------------------------
KERNAL_SIZE = 5 # Kernal size value for morphology operations.

def load_image(filename):
    """Import and return the image."""
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image

def binarize(image, threshold_value=150):
    """Convert and return the image in a binary format, using thresholding."""
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def dilation(binary_image, kernel_size=KERNAL_SIZE, iterations=1):
    """Apply and return the output of conducting a dilation operation on the binary image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=iterations)
    return dilated_image

def erosion(binary_image, kernel_size=KERNAL_SIZE, iterations=1):
    """Apply and return the output of conducting an erosion operation on the binary image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=iterations)
    return eroded_image

def opening(binary_image, kernel_size=KERNAL_SIZE):
    """Apply and return the output of conducting an opening operation on the binary image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return opened_image

def closing(binary_image, kernel_size=KERNAL_SIZE):
    """Apply and return the output of conducting a closing operation on the binary image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed_image

def display_images(original_image, binary_image, dilated_image, eroded_image, opened_image, closed_image):
    """Display the original, binary, and processed images, in a 3x2 grid."""
    # Primary display grid dimensions.
    width = 12
    height = 8
    plt.figure(
        figsize=(
            width, 
            height
        )
     )
    # Original section.
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    # Binary section.
    plt.subplot(2, 3, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    # Dilated section.
    plt.subplot(2, 3, 3)
    plt.imshow(dilated_image, cmap='gray')
    plt.title("Dilated Image")
    plt.axis('off')
    # Erroded section.
    plt.subplot(2, 3, 4)
    plt.imshow(eroded_image, cmap='gray')
    plt.title("Eroded Image")
    plt.axis('off')
    # Opened section.
    plt.subplot(2, 3, 5)
    plt.imshow(opened_image, cmap='gray')
    plt.title("Opened Image")
    plt.axis('off')
    # Closed section.
    plt.subplot(2, 3, 6)
    plt.imshow(closed_image, cmap='gray')
    plt.title("Closed Image")
    plt.axis('off')
    # Ending portion.
    plt.tight_layout() # Final display grid layout.
    plt.show() # Final display output.

def main():
    """Main driver function."""
    # Import and check if the image loaded correctly.
    original_image = load_image("handwritten_note.jpg")
    if original_image is None:
        print("ERROR: Image not found!")
        return
    binary_image = binarize(original_image) # Convert image to binary.
    # Apply morphological operations.
    dilated_image = dilation(binary_image)
    eroded_image = erosion(binary_image)
    opened_image = opening(binary_image)
    closed_image = closing(binary_image)
    # Display the original and processed images.
    display_images(
        original_image,
        binary_image,
        dilated_image,
        eroded_image,
        opened_image,
        closed_image
    )

# The big red activation button.
if __name__ == "__main__":
    main()  # Execute the main driver function.
