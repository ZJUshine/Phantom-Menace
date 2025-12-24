"""
EM Truncation Attack Module
This module implements an electromagnetic truncation attack on images.
"""
import numpy as np

def em_truncation(img_array, truncate_ratio=0.5):
    """
    Apply electromagnetic truncation attack to an image.
    
    Args:
        img_array: Numpy array of the image to be attacked
        
    Returns:
        Numpy array with EM truncation attack applied
    """
    # Calculate truncation parameters
    start_line_num = int(np.shape(img_array)[0] * truncate_ratio)
    line_num = int(np.shape(img_array)[0] * truncate_ratio)
    end_line_num = start_line_num + line_num
    
    # Remove middle section and append bottom section
    img1 = np.delete(img_array, slice(start_line_num, end_line_num), axis=0)
    img2 = np.append(img1, img_array[-line_num:], axis=0)
    
    return img2
