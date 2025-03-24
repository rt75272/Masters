import cv2 # type: ignore
# ---------------------------------------------------------------------------------------------
# OpenCV Smoke Test
# 
# Imports an image, displays the image for a given amount of time, saves a copy of the image
# in the user's current directory, and automatically closes the window displaying the image.
#
# Usage:
#   $ python cv_smoketest.py
# ---------------------------------------------------------------------------------------------
def import_image(filename):
    """Import the image."""
    imported_image = cv2.imread(filename)
    return imported_image

def display_image(image_filename):
    """Display the image for a certain amount of time."""
    wait_time = 2000 # Amount of time(milliseconds) the image is displayed.
    cv2.imshow("Brain Image", image_filename) # Show the image.
    cv2.waitKey(wait_time) # Display the image for a given amount of time.

def save_image_copy(image_input):
    """Save a copy of the image in the current directory."""
    output_path = './brain_image_copy.jpg' # String to save the copy in the current directory.
    cv2.imwrite(output_path, image_input) # Save image copy.
    print(f"Image saved to {output_path}")

def main():
    """Main driver function."""
    image_file = "brain_image.jpg" # Input image filename.
    test_image = import_image(image_file)
    # Check if the image loaded correctly.
    if test_image is None:
        print("ERROR: Image not found!")
    else:
        display_image(test_image)
        save_image_copy(test_image) 
        cv2.destroyAllWindows() # Close all OpenCV GUIs.
        print("Program Completed!") # Final display message.

# Big red activation button.
if __name__ == "__main__":
    main() # Execute the main driver function.