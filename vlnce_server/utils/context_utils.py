from PIL import Image
import numpy as np

def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
    """Convert a numpy array to a PIL RGB image."""
    if numpy_array.shape[-1] == 3:
        # Convert numpy array to RGB PIL Image
        return Image.fromarray(numpy_array, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")

