import tensorflow as tf
"""GPU configuration utilities for TensorFlow."""

def configure_gpu():
    """Configure TensorFlow to use GPU with memory growth, fallback to CPU if 
    unavailable."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for each GPU.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Set the first GPU as the default device.
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            print(f"GPU configured successfully. Using GPU: {gpus[0]}")
            print(f"Available GPUs: {len(gpus)}")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            print("Falling back to CPU execution.")
            return False
    else:
        print("No GPU found. Running on CPU.")
        print("To use GPU, ensure CUDA drivers and libraries are properly installed.")
        print("See: https://www.tensorflow.org/install/gpu")
        return False
