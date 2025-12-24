"""
EM Color Strip Attack Module
This module implements an electromagnetic color strip attack on images.
"""
import numpy as np

def em_color_strip(img_array, num_stripes=12):
    """
    Apply electromagnetic color strip attack to an image.
    
    Args:
        img_array: Numpy array of the image to be attacked
        
    Returns:
        Numpy array with EM color strip attack applied
    """
    # Create a copy of the input array to avoid modifying the original
    result_array = np.copy(img_array)
    
    height, width = img_array.shape[:2]
    step = int(height/num_stripes)
    
    for x in range(0, width):
        for down in range(0, height-step, 2*step):
            up = down + step
            for y in range(down, up):
                # Apply color transformation
                g_value = img_array[y, x, 1]
                r_value = img_array[y, x, 0]
                b_value = img_array[y, x, 2]
                
                # Calculate new values
                new_r = int(g_value * 2.5)
                new_g = int((r_value + b_value) / 2) - 50
                new_b = int(g_value * 2.5)
                
                # Clip values to valid range [0, 255]
                new_r = max(0, min(255, new_r))
                new_g = max(0, min(255, new_g))
                new_b = max(0, min(255, new_b))
                
                # Update pixel in result array
                result_array[y, x, 0] = new_r
                result_array[y, x, 1] = new_g
                result_array[y, x, 2] = new_b
    
    return result_array
