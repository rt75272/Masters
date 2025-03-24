import cv2  # type: ignore
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------
# Image Segmentation Thresholding.
# 
# Demonstrates the use of adaptive thresholding to segment different types of images including
# an indoor scene, outdoor scenery, and a close-up image of a single object. The images are 
# processed using adaptive thresholding techniques, and the results are displayed for 
# comparison.
# 
# Usage:
#   $ python thresholding.py
# ---------------------------------------------------------------------------------------------
def load_image(filename):
    """Import and return the image in grayscale."""
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image

def adaptive_threshold(image, block_size=11, C=2):
    """Apply and return the output of using adaptive thresholding on the input image."""
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, C
    )

def display_images(original_image, thresholded_image, image_title):
    """Display the original and thresholded images side by side."""
    # Primary display grid dimensions.
    width = 12
    height = 8
    fig, axes = plt.subplots(1,2,
        figsize=(width,height))
    # Original section.
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'{image_title} - Original')
    axes[0].axis('off')
    # Original section.
    axes[1].imshow(thresholded_image, cmap='gray')
    axes[1].set_title(f'{image_title} - Thresholded')
    axes[1].axis('off')
    # Ending portion.
    plt.tight_layout() # Final display grid layout.
    plt.show() # Final display output.

def main():
    """Main driver function."""
    # Load the images.
    indoor_scene = load_image("indoor_scene.jpg")
    outdoor_scene = load_image("outdoor_scene.jpg")
    close_up_object = load_image("close_up_object.jpg")
    # Implement adaptive thresholding on each image.
    indoor_scene_thresh = adaptive_threshold(indoor_scene)
    outdoor_scene_thresh = adaptive_threshold(outdoor_scene)
    close_up_object_thresh = adaptive_threshold(close_up_object)
    # Display the original and recently thresholded images.
    display_images(indoor_scene, indoor_scene_thresh, "Indoor Scene")
    display_images(outdoor_scene, outdoor_scene_thresh, "Outdoor Scene")
    display_images(close_up_object, close_up_object_thresh, "Close-Up Object")

# The big red activation button.
if __name__ == "__main__":
    main()  # Execute the main driver function.
