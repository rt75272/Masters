import cv2 # type: ignore
# ---------------------------------------------------------------------------------------------
# Face Detection and Eye Blurring
# 
# Demonstrates the use of face detection and anonymizing features by applying Gaussian blurring
# to the eyes of detected faces in color images. The algorithm detects faces using cascade
# classifiers and then processes the eyes within the detected face regions to blur them out for 
# privacy concerns.
#
# Usage:
#   $ python face_blurring.py
# ---------------------------------------------------------------------------------------------
ZERO = 0

def process_image(image_path, output_path):
    """Detect faces and blur eyes for privacy."""
    # Load the cascades for the face and eye detection.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # Load the image and convert to grayscale.
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Enhance the image for better face detection.
    gray_image = cv2.equalizeHist(gray_image) # Improve contrast.
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), ZERO) # Reduce noise.
    # Detect faces in the grayscale image.
    faces = face_cascade.detectMultiScale(
                                        gray_image,
                                        scaleFactor=1.1, 
                                        minNeighbors=5, 
                                        minSize=(30, 30))
    # Iterate over the faces detected and draw bounding boxes around them.
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (ZERO, ZERO, 255), 2) # Red rectangle around located face(s).
        # Extract the face region ROI for eye detection.
        face_roi = gray_image[y:y + h, x:x + w]
        color_face_roi = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_roi) # Detect eyes within the face region.
        # Iterate over the eyes detected and blur them.
        for (ex, ey, ew, eh) in eyes:
            eye_roi = color_face_roi[ey:ey + eh, ex:ex + ew] # Extract the eye region from the face region.
            blurred_eye = cv2.GaussianBlur(eye_roi, (15, 15), ZERO) # Apply Gaussian blur to the eye.
            color_face_roi[ey:ey + eh, ex:ex + ew] = blurred_eye # Replace eye region with blurred version.
    cv2.imwrite(output_path, image) # Save the processed image with blurred eyes.
    print(f"Processed image saved as {output_path}")
    cv2.imshow('Processed Image', image) # Display the processed image with blurred eyes
    # Wait until the window is closed by the user.
    while True:
        # Check if the window was closed.
        if cv2.getWindowProperty('Processed Image', cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.waitKey(1) # A small time to refresh the window.
    cv2.destroyAllWindows() # Close the image window after the user closes it

def main():
    """Main driver function. Processes multiple images."""
    images = [
        ("whole_body_person.jpg", "single_person_output.jpg"),
        ("multiple_people.jpg", "multiple_people_output.jpg"),
        ("with_dog.jpg", "dog_output.jpg"),
    ]
    # Process each image in the list of images.
    for image_path, output_path in images:
        process_image(image_path, output_path)

# The big red activation button
if __name__ == "__main__":
    main()
