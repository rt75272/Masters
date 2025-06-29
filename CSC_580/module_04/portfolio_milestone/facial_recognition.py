import face_recognition # type: ignore
import cv2 # type: ignore
# --------------------------------------------------------------------------------------
# Facial Recognition.
#
# Checks if a person appears in a group photo by comparing face encodings. Uses the
# facial_recognition library.
#
# Usage:
#     $ python facial_recognition.py
# --------------------------------------------------------------------------------------
def load_and_encode_person_image(person_image_path):
    """Loads and encodes a person's face from an image file."""
    # Loads the image file into a numpy array format for processing.
    person_image = face_recognition.load_image_file(person_image_path)
    # Extracts face encodings (128-dimensional vectors) from the image. This creates a
    # unique numerical representation of each face found.
    person_encodings = face_recognition.face_encodings(person_image)
    # Checks if any faces were detected in the image.
    if len(person_encodings) == 0:
        print("No face found in the person's image.")
        return False, None    
    # Uses the first face found.
    person_encoding = person_encodings[0]
    return True, person_encoding

def load_and_detect_group_faces(group_image_path):
    """Loads group image and detect all faces in it."""
    # Loads the group photo into memory for processing.
    group_image = face_recognition.load_image_file(group_image_path)
    # Finds the pixel coordinates of all faces in the group image. Returns (top, right,
    # bottom, left) coordinates for each face.
    group_face_locations = face_recognition.face_locations(group_image)
    # Generates face encodings for each detected face using their locations. This creates
    # unique identifiers for comparison with the target person.
    group_encodings = face_recognition.face_encodings(group_image, 
                                                      group_face_locations)
    return group_image, group_face_locations, group_encodings

def compare_faces(group_encodings, person_encoding, tolerance=0.6):
    """Compares person's face encoding with group face encodings."""
    # Compares the target person's encoding against each face in the group. Returns a
    # list of True/False values for each face comparison.
    return face_recognition.compare_faces(group_encodings, person_encoding,
                                          tolerance=tolerance)

def visualize_results(group_image, face_locations, match_results):
    """Displays the group image with rectangles around faces."""
    # Converts from RGB (face_recognition format) to BGR (OpenCV format).
    group_image_bgr = cv2.cvtColor(group_image, cv2.COLOR_RGB2BGR)
    # Draws rectangles around each detected face.
    for (top, right, bottom, left), match in zip(face_locations, match_results):
        # Green rectangle for matches, red for non-matches.
        color = (0, 255, 0) if match else (0, 0, 255)  
        # Draws rectangle with 2-pixel thickness.
        cv2.rectangle(group_image_bgr, (left, top), (right, bottom), color, 2)
    try:
        # Displays the annotated image in a window.
        cv2.imshow('Group Image Check', group_image_bgr)
        print("Checking if the individual is in the group photo...")
        print("Green rectangles = matches, Red rectangles = no matches")
        # Waits for 5 seconds or until a key is pressed.
        key = cv2.waitKey(5000)
        # Cleans up the display window.
        cv2.destroyAllWindows()
    except Exception as e:
        # Handles display errors.
        print(f"Display error: {e}")
        print("Continuing without displaying image...")
        cv2.destroyAllWindows()

def is_person_in_group(person_image_path, group_image_path, tolerance=0.6, 
                       show_visualization=True):
    """Checks if a person appears in a group photo."""
    # Loads and encodes the individual's image.
    success, person_encoding = load_and_encode_person_image(person_image_path)
    if not success:
        return False
    # Loads and detects faces in the group image.
    group_image, group_face_locations, group_encodings = \
        load_and_detect_group_faces(group_image_path)
    # Verifies that faces were found in the group image.
    if len(group_encodings) == 0:
        print("No faces found in the group image.")
        return False
    # Compares the person's face against all faces in the group.
    results = compare_faces(group_encodings, person_encoding, tolerance)
    # Displays visual results if requested.
    if show_visualization:
        visualize_results(group_image, group_face_locations, results)
    # Returns True if the person was found in any of the group faces.
    return any(results)

def main():
    """Main driver function."""
    person_path = "person.jpg"
    group_path = "group.jpg"
    print("Starting facial recognition analysis...")
    print(f"Person image: {person_path}")
    print(f"Group image: {group_path}")
    # Executes the main facial recognition logic.
    found = is_person_in_group(person_path, group_path)
    # Displays the final result to the user.
    if found:
        print("The person is in the group photo.")
    else:
        print("The person is NOT in the group photo.")

# The big red activation button.
if __name__ == "__main__":
    main()
