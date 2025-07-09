import os
"""Configuration settings for the Cat vs Dog Classifier."""

# Image processing settings.
IMG_SIZE = (128, 128) # Image size for resizing all images.
SAMPLE_SIZE = 4096 # Number of training images to use.
VALID_SIZE = 1024 # Number of validation images to use.

# Training settings.
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 5e-2
DENSE_LEARNING_RATE = 1e-2

# Model architecture settings.
DENSE_FC_SIZE = 512
CNN_FC_SIZE = 512

# Data directories.
DATA_DIR = 'data'
TRAIN_CAT_DIR = os.path.join(DATA_DIR, 'train', 'cats')
TRAIN_DOG_DIR = os.path.join(DATA_DIR, 'train', 'dogs')
TEST_CAT_DIR = os.path.join(DATA_DIR, 'test', 'cats')
TEST_DOG_DIR = os.path.join(DATA_DIR, 'test', 'dogs')

# Model save paths.
MODEL_SAVE_PATH = 'conv_model_big.keras'
CHECKPOINT_PATH = 'best_model_checkpoint.keras'

# GPU settings.
SUPPRESS_TF_WARNINGS = True
TF_LOG_LEVEL = '3'  # 0=all_messages, 1=info, 2=warning, 3=error.
