import os
import glob
import numpy as np
import concurrent.futures
from collections import defaultdict
from PIL import Image
from config import IMG_SIZE
"""Data loading and preprocessing utilities."""
class DataLoader:
    """Handles loading and preprocessing of image data."""
    def __init__(self):
        """Constructor for DataLoader."""
        pass
    
    def pixels_from_path(self, file_path):
        """Loads an image from a file path, resizes it, normalizes it, and returns it as
          a numpy array."""
        im = Image.open(file_path)
        im = im.resize(IMG_SIZE)  # PIL resize uses (width, height) format
        img_array = np.array(im)
        # Handle grayscale images
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        return img_array
    
    def analyze_image_shapes(self, image_dir, sample_count=1000):
        """Analyzes the shapes of the first sample_count images in the given directory.
        Prints progress every 100 images. Returns a dictionary with shape counts."""
        shape_counts = defaultdict(int)
        image_paths = glob.glob(os.path.join(image_dir, '*'))[:sample_count]
        
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processed {i} images")
            shape_counts[str(self.pixels_from_path(image_path).shape)] += 1
        return shape_counts
    
    def load_images_parallel(self, paths):
        """Helper to load images in parallel and reduce execution time."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            images = list(executor.map(self.pixels_from_path, paths))
        images = np.asarray(images)
        return images
    
    def load_dataset(self, cat_dir, dog_dir, sample_size):
        """Loads a balanced set of cat and dog images."""
        # Check if directories exist
        if not os.path.exists(cat_dir):
            print(f"ERROR: Cat directory '{cat_dir}' does not exist!")
            return np.array([]), np.array([])
        if not os.path.exists(dog_dir):
            print(f"ERROR: Dog directory '{dog_dir}' does not exist!")
            return np.array([]), np.array([])
        
        cat_paths = glob.glob(os.path.join(cat_dir, '*'))
        dog_paths = glob.glob(os.path.join(dog_dir, '*'))
        
        print(f"Found {len(cat_paths)} cat images in {cat_dir}")
        print(f"Found {len(dog_paths)} dog images in {dog_dir}")
        
        if len(cat_paths) == 0 or len(dog_paths) == 0:
            print("ERROR: No images found in one or both directories!")
            return np.array([]), np.array([])
        
        min_count = min(len(cat_paths), len(dog_paths), sample_size)
        cat_paths = cat_paths[:min_count]
        dog_paths = dog_paths[:min_count]
        
        print(f"Loading {min_count} images from each category...")
        cat_set = self.load_images_parallel(cat_paths)
        dog_set = self.load_images_parallel(dog_paths)
        
        if len(cat_set) == 0 or len(dog_set) == 0:
            print("ERROR: Failed to load images!")
            return np.array([]), np.array([])
        
        x = np.concatenate([cat_set, dog_set])
        y = np.asarray([1]*min_count + [0]*min_count)
        
        # Shuffle the dataset
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        return x, y
    
    def validate_data_directories(
            self, 
            train_cat_dir, 
            train_dog_dir, 
            test_cat_dir, 
            test_dog_dir):
        """Validate that all required data directories exist."""
        directories = [train_cat_dir, train_dog_dir, test_cat_dir, test_dog_dir]
        dir_names = ['Training Cats', 'Training Dogs', 'Test Cats', 'Test Dogs']
        missing_dirs = []
        for dir_path, dir_name in zip(directories, dir_names):
            if not os.path.exists(dir_path):
                missing_dirs.append((dir_path, dir_name))
        if missing_dirs:
            print("ERROR: Missing data directories:")
            for dir_path, dir_name in missing_dirs:
                print(f"  - {dir_name}: {dir_path}")
            print("\nExpected directory structure:")
            print("data/")
            print("├── train/")
            print("│   ├── cats/")
            print("│   └── dogs/")
            print("└── test/")
            print("    ├── cats/")
            print("    └── dogs/")
            return False
        return True
