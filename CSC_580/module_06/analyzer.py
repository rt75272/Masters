import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
"""Analysis and visualization utilities."""
class ModelAnalyzer:
    """Handles model analysis and result visualization."""
    def __init__(self):
        """Constructor for ModelAnalyzer."""
        pass

    def scatterplot_analysis(self, preds, labels_valid):
        """Plots a scatterplot of predictions vs. true labels and analyzes 
        thresholds."""
        sns.scatterplot(x=preds, y=labels_valid)
        print("\nThreshold Analysis (for classifying as 'cat'):")
        print("Threshold | Accuracy | Precision")
        print("-" * 30)
        for i in range(1, 10):
            threshold = 0.1 * i
            selected = labels_valid[preds > threshold]
            if len(selected) > 0:
                acc = sum(selected) / len(selected)
            else:
                acc = 0
            print(f"   {threshold:.1f}    |  {acc:.3f}   |  {acc:.3f}")
        print("-" * 30)
    
    def generate_confusion_matrix(self, model, x_valid, labels_valid):
        """Generate and display a formatted confusion matrix."""
        pred_probs = model.predict(x_valid)
        if len(pred_probs.shape) > 1 and pred_probs.shape[1] == 1:
            pred_probs = pred_probs.flatten()
        pred_labels = (pred_probs > 0.5).astype(int)
        cm = confusion_matrix(labels_valid, pred_labels)
        print("Confusion Matrix:")
        print("         Predicted")
        print("         Dog  Cat")
        print(f"Actual Dog {cm[0][0]:4} {cm[0][1]:4}")
        print(f"     Cat {cm[1][0]:4} {cm[1][1]:4}")
        print(f"\nAccuracy: {(cm[0][0] + cm[1][1]) / cm.sum():.3f}")
        print(f"Total predictions: {cm.sum()}")
        return cm
    
    def get_prediction(self, model, x_valid, index):
        """Returns the predicted probability that the image at the given index is a 
        cat."""
        img_array = np.asarray([x_valid[index]])
        prediction = model.predict(img_array)
        # Handle both (n,) and (n, 1) output shapes
        if len(prediction.shape) > 1 and prediction.shape[1] == 1:
            prob = prediction[0][0]
        else:
            prob = prediction[0]
        return prob
    
    def get_image_for_display(self, x_valid, index):
        """Returns a PIL Image object for display."""
        img_array = x_valid[index]
        # Convert normalized values back to 0-255 range for display
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        return img
    
    def format_prediction_result(self, index, probability, true_label):
        """Format prediction results for display."""
        label_name = "cat" if true_label == 1 else "dog"
        result = (f"Index {index}: "
                 f"Probability of being a cat: {probability:.2f} "
                 f"(Label: {label_name})")
        return result
    
    def display_image_silently(self, img):
        """Displays an image without printing to stderr."""
        with open(os.devnull, 'w') as f, os.fdopen(os.dup(2), 'w') as old_stderr:
            os.dup2(f.fileno(), 2)
            img.show()
            os.dup2(old_stderr.fileno(), 2)
