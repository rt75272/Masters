import tensorflow as tf
from config import TF_LOG_LEVEL
"""GPU configuration and setup utilities."""
class GPUManager:
    """Manages GPU configuration and monitoring."""
    def __init__(self):
        """Constructor for GPUManager."""
        self.gpu_available = False
        self.setup_gpu()
    
    def setup_gpu(self):
        """Configure GPU settings for optimal performance."""
        print("TensorFlow version:", tf.__version__)
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        if tf.config.list_physical_devices('GPU'):
            self.gpu_available = True
            print("GPU devices found:")
            for gpu in tf.config.list_physical_devices('GPU'):
                print(f"  {gpu}")
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory.
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth enabled")
                except RuntimeError as e:
                    print(f"GPU memory growth setting failed: {e}")
            # Set GPU as the default device.
            gpu_devices = tf.config.list_physical_devices('GPU')
            first_gpu = gpu_devices[0]
            tf.config.set_visible_devices(first_gpu, 'GPU')
            print("GPU set as default device")
        else:
            print("No GPU found. Using CPU.")
    
    def check_gpu_usage(self):
        """Check if GPU is being used and print device information."""
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        print("Current device context:", tf.config.get_visible_devices())
        if tf.config.list_physical_devices('GPU'):
            print("Using GPU for computations")
            return True
        else:
            print("Using CPU for computations")
            return False
    
    def get_device_context(self):
        """Get the appropriate device context string."""
        return '/GPU:0' if self.gpu_available else '/CPU:0'
    
    def ensure_gpu_tensors(self, x, y):
        """Ensure tensors are on GPU if available."""
        if self.gpu_available:
            with tf.device('/GPU:0'):
                x_gpu = tf.constant(x)
                y_gpu = tf.constant(y)
                return x_gpu, y_gpu
        return x, y
