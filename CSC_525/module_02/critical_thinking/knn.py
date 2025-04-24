import sys
import warnings
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
# ------------------------------------------------------------------------------------------------------------
# KNN Game Genre Classifier.
#
# Implements a KNN classifier in order to predict a person's favorite video game genre using the following 
# features: age, height, weight, and gender.
#
# Usage:
#   $ python knn_predict.py [age] [height] [weight] [gender]
#       - age: In years.
#       - height: In inches.
#       - weight: In pounds.
#       - gender: 0 for female, 1 for male.
#       --- If no arguments are given, the default argument values will be used.
# ------------------------------------------------------------------------------------------------------------
DEFAULT_INPUT = [25.0, 68.0, 150.0, 1] # Default user input values.
K_NEIGHBORS = 10 # Number of neighbors to use in KNN classifier.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn") # Suppress warnings from sklearn.

def load_data():
    """Loads, prepares, and returns the dataset."""
    data = pd.read_csv('data.csv') # Read the dataset from the CSV file.
    data.columns = ['Age', 'Height', 'Weight', 'Gender', 'Genre'] # Set column names.
    return data

def train_knn_classifier(data):
    """Trains and returns the KNN model, along with the necessary preprocessing tools."""
    # Extract features(age, height, weight, gender) and the label(genre) from the dataset.
    features = data[['Age', 'Height', 'Weight', 'Gender']] # Features.
    label = data['Genre'] # Target.
    # Convert genre labels into numeric values(ex.'Action'-->0,'Adventure'-->1).
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label)
    # Scale the features in order to standardize the range.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    # Initialize and train the KNN classifier.
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights='distance')
    knn.fit(X_scaled, y_encoded)
    return knn, scaler, label_encoder

def parse_input_arguments():
    """Parses command line input or uses the default values."""
    while True: # Loop to allow for re-entry of input if invalid argument data types are given.
        if len(sys.argv) == 5: # Check for the correct number of arguments.
            try: 
                return [float(arg) for arg in sys.argv[1:]] # Convert all input to float data types.
            except ValueError: # Catch invalid data types and attempt to have user enter arguments again.
                print("Error: Please provide valid numeric inputs for arguments, age, height, weight, and gender.")
                print("Please re-enter valid numeric values for the arguments.")
                sys.argv[1:] = input("Enter aruguments, age, height, weight, gender: ").split() # Prompt user to reenter arguments.
        else: # If no or incomplete input is given, use the specified default values.
            print("No or incomplete input detected.\n\tUsing default argument values: Age=33, Height=62, Weight=180, Gender=0(female)\n")
            return DEFAULT_INPUT

def predict_genre(knn, scaler, label_encoder, input_data):
    """Scales input data and returns the predicted genre."""
    scaled_input = scaler.transform([input_data]) # Scale user input data.
    prediction = knn.predict(scaled_input)[0] # Use the trained KNN classifier to predict the genre.
    prediction = label_encoder.inverse_transform([prediction])[0] # Convert predicted index back to a genre label.
    return prediction

def main():
    """Main driver function in order to load the data, train the KNN model, and make a prediction."""
    data = load_data() # Load and prepare data.
    knn, scaler, label_encoder = train_knn_classifier(data) # Train the KNN model and get the necessary preprocessing tools.
    user_input = parse_input_arguments() # Parse the user's input, or use the specified defaults.
    genre = predict_genre(knn, scaler, label_encoder, user_input) # Predict the genre based on the user's input data.
    print(f"Predicted Favorite Game Genre: {genre}") # Output the predicted genre to the terminal.

# The big red activation button.
if __name__ == "__main__":
    main()
