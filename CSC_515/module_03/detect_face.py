import cv2 # type: ignore
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------
# Face and Eyes Object Detection. 
# 
# Detects faces and eyes in an image, draws red bounding boxes around the eyes, places a green 
# circle around the detected face(s), and adds text "This is me?".
#
# Usage:
#   $ python detect_face.py
# ---------------------------------------------------------------------------------------------
MAX = 255 # Maximum RGB color value.

def import_image(filename):
    """Import and return the image."""
    image = cv2.imread(filename)
    return image

def detect_faces(image):
    """Detect and return faces in the given image and return the list of bounding boxes."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def detect_eyes(image, face_region):
    """Detect and return eyes within a face region and return the list of bounding boxes."""
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = image[face_region[1]:face_region[1]+face_region[3], face_region[0]:face_region[0]+face_region[2]]
    eyes = eye_cascade.detectMultiScale(gray)
    return eyes

def draw_face_circle(image, face_region):
    """Place a green circle around the face."""
    x, y, w, h = face_region
    radius = max(w, h) // 2
    center = (x + w // 2, y + h // 2)
    cv2.circle(image, center, radius, (0, MAX, 0), 5)  # Green circle.

def draw_eye_boxes(image, face_region, eyes):
    """Draw a pair of red bounding boxes around the eyes."""
    x, y, w, h = face_region
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, MAX), 2)  # Red bounding boxes.

def add_text(image, text, position=(50, 50)):
    """Add given text to the image."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (MAX, MAX, MAX), 2)

def display_image(image, title):
    """Display the provided image using matplotlib."""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def main():
    """Main driver function."""
    # Import the image and check if the image loaded correctly.
    image = import_image("selfie.jpg")
    if image is None:
        print("ERROR: Image not found!")
        return
    # Detect faces and loop over each detected face.
    faces = detect_faces(image)
    for face in faces:
        draw_face_circle(image, face) # Draw a green circle around the face.
        eyes = detect_eyes(image, face) # Detect eyes within the face region.
        draw_eye_boxes(image, face, eyes) # Draw red bounding boxes around the eyes.
    add_text(image, "This is me?") # Add text as the title on the image.
    display_image(image, "Adjusted Image")  # Display the final image.
    print("Program Completed!") # Output the final message.

# Big red activation button.
if __name__ == "__main__":
    main()  # Execute the main driver function.
