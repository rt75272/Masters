from PIL import Image, ImageDraw
import face_recognition  # type: ignore
# ------------------------------------------------------------------------------------- 
# Face Detection.
#
# Loads an image file, detects faces in it, and draws green bounding boxes around the 
# detected faces.
#
# Usage:
#   python face_detect.py
# -------------------------------------------------------------------------------------
def load_image(image_path):
    """Loads and returns an image file as a numpy array using face_recognition."""
    image = face_recognition.load_image_file(image_path)
    return image

def find_faces(image):
    """Detects faces in the image and returns their locations as a list."""
    face_locations = face_recognition.face_locations(image)
    return face_locations

def draw_boxes(image, face_locations):
    """Draws green bounding boxes around detected faces and returns the PIL image."""
    pil_image = Image.fromarray(image)# Convert the array image to a PIL Image object.
    draw_handle = ImageDraw.Draw(pil_image) # Create a drawing context.
    # Loop through each detected face location.
    for face_location in face_locations:
        top, right, bottom, left = face_location # Unpack the coordinates of the face.
        # Print the coordinates of the detected face.
        print("\nA face is located at pixel location:")
        print("  Top: {}".format(top))
        print("  Left: {}".format(left))
        print("  Bottom: {}".format(bottom))
        print("  Right: {}".format(right))
        # Place a green bounding box around the detected face.
        draw_handle.rectangle([left, top, right, bottom], outline="green", width=7)
    return pil_image

def main():
    # Load the image file.
    image_path = "input_image.jpg"
    image = load_image(image_path)
    print("\nImage loaded successfully from {}".format(image_path))
    # Detect faces in the image and draw bounding boxes.
    print("\nDetecting faces in the image...")
    face_locations = find_faces(image) # Detect faces in the image.
    n_faces = len(face_locations) # Count the number of detected faces.
    pil_image = draw_boxes(image, face_locations) # Draw bounding boxes around faces.
    print("\nFound {} face(s) in this picture.".format(n_faces))
    print("\nLoading adjusted image with red bounding boxes...")
    pil_image.show() # Display the image with bounding boxes.

# The big red activation button.
if __name__ == "__main__":
    main()