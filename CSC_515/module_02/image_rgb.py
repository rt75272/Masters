import cv2  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------
# RGB Channel Manipulation
# 
# Loads an image, extracts its RGB channels, displays the individual channels, merges them back
# into a color image, and swaps the Red and Green channels (GRB).
#
# Usage:
#   $ python image_rgb.py
# ---------------------------------------------------------------------------------------------
def import_image(filename):
    """Import and return the image."""
    image = cv2.imread(filename)
    return image

def extract_channels(image):
    """Extract and return individual RGB channels from the imported image."""
    blue_channel = image[:,:,0] # Blue.
    green_channel = image[:,:,1] # Green.
    red_channel = image[:,:,2] # Red.
    return blue_channel, green_channel, red_channel

def display_channels(blue_channel, green_channel, red_channel):
    """Display the individual channels (Blue, Green, Red)."""
    plt.figure(figsize=(10, 10)) # Display window size.
    # Blue section.
    plt.subplot(1, 3, 1)
    plt.imshow(blue_channel, cmap="Blues")
    plt.title("Blue Channel")
    # Green section.
    plt.subplot(1, 3, 2)
    plt.imshow(green_channel, cmap="Greens")
    plt.title("Green Channel")
    # Red section.
    plt.subplot(1, 3, 3)
    plt.imshow(red_channel, cmap="Reds")
    plt.title("Red Channel")
    plt.show() # Display the sections/channels.

def merge_channels(blue_channel, green_channel, red_channel):
    """Merge and return the RGB channels back into a 3D color image."""
    merged_bgr = cv2.merge([blue_channel, green_channel, red_channel])
    return merged_bgr

def swap_red_green(blue_channel, red_channel, green_channel):
    """Swap and return the Red and Green channels (GRB)."""
    swapped_bgr = cv2.merge([blue_channel, red_channel, green_channel])
    return swapped_bgr

def display_image(image, titlename):
    """Display the merged or manipulated image."""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for correct display.
    plt.title(titlename)
    plt.show()

def main():
    """Main driver function."""
    image_file = "puppy_image.jpg"  # Input image filename.
    image = import_image(image_file) # Import the image.
    # Check if the image loaded correctly.
    if image is None:
        print("ERROR: Image not found!")
        return
    # Extract blues, greens, and reds, section.
    blue_channel, green_channel, red_channel = extract_channels(image) # Extract channels from the image.
    display_channels(blue_channel, green_channel, red_channel) # Display the individual channels.
    # Merge back to original section.
    merged_bgr = merge_channels(blue_channel, green_channel, red_channel) # Merge the channels back into a 3D image (BGR).
    display_image(merged_bgr, "Original - Merged Back Together") # Display the original merged image.
    # Swap GRB section.
    swapped_bgr = swap_red_green(blue_channel, red_channel, green_channel) # Swap Red and Green channels (GRB).
    display_image(swapped_bgr, "Swapped - GRB") # Display the image with swapped channels.
    # Done section.
    print("Program Completed!") # Final display message.

# Big red activation button.
if __name__ == "__main__":
    main()  # Execute the main driver function.
