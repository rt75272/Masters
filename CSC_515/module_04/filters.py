import cv2  # type: ignore
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------
# Image Filtering
# 
# Applies mean, median, and Gaussian filters using various kernel sizes and sigma values. Then
# displays the filtered images in a side-by-side comparison grid.
# 
# Usage:
#   $ python filters.py
# ---------------------------------------------------------------------------------------------
def load_image(filename):
    """Import and return the image."""
    image = cv2.imread(filename)
    return image

def mean_filter(image, kernel_size):
    """Implement and return a mean filter on the image with a given kernel size."""
    return cv2.blur(image, (kernel_size, kernel_size))

def median_filter(image, kernel_size):
    """Implement and return a median filter on the image with a given kernel size."""
    return cv2.medianBlur(image, kernel_size)

def gaussian_filter(image, kernel_size, sigma):
    """Implement and return a Gaussian filter on the image with a given kernel size and sigma value."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def display_filters(image, sigma_values):
    """Display the filtered images in a 3x4 grid."""
    kernel_sizes = [3, 5, 7]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    # Loop over kernel sizes.
    for i, kernel_size in enumerate(kernel_sizes):
        # Apply filters for the current kernel size.
        mean_filtered = mean_filter(image, kernel_size)
        median_filtered = median_filter(image, kernel_size)
        gaussian_filtered_1 = gaussian_filter(image, kernel_size, sigma_values[0])
        gaussian_filtered_2 = gaussian_filter(image, kernel_size, sigma_values[1])
        # Display the filtered images in the matplotlib subplot grid.
        axes[i, 0].imshow(cv2.cvtColor(mean_filtered, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Mean Filter {kernel_size}x{kernel_size}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f'Median Filter {kernel_size}x{kernel_size}')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(cv2.cvtColor(gaussian_filtered_1, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f'Gaussian Filter {kernel_size}x{kernel_size} (sigma={sigma_values[0]})')
        axes[i, 2].axis('off')
        axes[i, 3].imshow(cv2.cvtColor(gaussian_filtered_2, cv2.COLOR_BGR2RGB))
        axes[i, 3].set_title(f'Gaussian Filter {kernel_size}x{kernel_size} (sigma={sigma_values[1]})')
        axes[i, 3].axis('off')
    # Display the adjusted images.
    plt.tight_layout()
    plt.show()

def main():
    """Main driver function."""
    # Import and check if the image loaded correctly.
    image = load_image("Mod4CT1.jpg")
    if image is None:
        print("ERROR: Image not found!")
        return
    sigma_values = [0.8, 2.0]  # Sigma values for different blurring effects.
    display_filters(image, sigma_values) # Display the filtered images.

# The big red activation button.
if __name__ == "__main__":
    main()  # Execute the main driver function.
