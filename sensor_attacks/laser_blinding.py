"""
Laser Blinding Attack Module
This module implements a laser blinding attack on images.
"""
import numpy as np
import math
from PIL import Image

def blooming(img_array, strength):
    """
    Apply blooming effect to an image.
    
    Args:
        img_array: Numpy array of the image
        strength: Intensity of the blooming effect
        
    Returns:
        Numpy array with blooming effect applied
    """
    rows, cols = img_array.shape[:2]
    centerX = rows / 2 - 0
    centerY = cols / 2 + 0
    radius = min(centerX, centerY)
    dst = np.zeros((rows, cols, 3), dtype="uint8")
    
    for i in range(rows):
        for j in range(cols):
            distance = math.pow((centerY-j), 2) + math.pow((centerX-i), 2)
            B = int(img_array[i,j][0])
            G = int(img_array[i,j][1])
            R = int(img_array[i,j][2])
            
            if (distance < radius * radius):
                result = int(strength * (1.0 - math.sqrt(distance) / radius))
                B = int(img_array[i,j][0]) + result
                G = int(img_array[i,j][1]) + result
                R = int(img_array[i,j][2]) + result
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                dst[i,j] = np.array([B, G, R], dtype=np.uint8)
            else:
                dst[i,j] = np.array([B, G, R], dtype=np.uint8)
    
    return dst

def adjust_exposure(image_array, exposure_value):
    """
    Adjust the exposure of an image.
    
    Args:
        image_array: Numpy array of the image
        exposure_value: Exposure adjustment factor
        
    Returns:
        Numpy array with adjusted exposure
    """
    # Convert the image to the float type
    image = image_array.astype(np.float32)
    
    # Adjust the exposure value
    image = exposure_value * image
    
    # Clip the pixel values that exceed the maximum value of 255
    image = np.clip(image, 0, 255)
    
    # Convert the image back to the uint8 type
    image = image.astype(np.uint8)
    
    return image

def over_exposure(image_array):
    """
    Apply over-exposure effect to an image.
    
    Args:
        image_array: Numpy array of the image
        
    Returns:
        Numpy array with over-exposure effect applied
    """
    exposure_value = 20  # exposure brightness
    strength = 300  # blooming center brightness
    
    image = adjust_exposure(image_array, exposure_value)
    image = blooming(image, strength)
    
    return image

def laser_blinding(img_array, laser_pattern_path, alpha=0.85):
    """
    Apply laser blinding attack by blending two images.
    
    Args:
        img_array: First image as numpy array
        laser_pattern_path: Path to the second image file
        alpha: Blending factor between 0 and 1
        
    Returns:
        Numpy array with laser blinding attack applied
    """
    laser_pattern_array = np.array(Image.open(laser_pattern_path).convert("RGB"))
    # Resize if necessary
    if img_array.shape[:2] != laser_pattern_array.shape[:2]:
        img1 = Image.fromarray(img_array)
        laser_pattern = Image.fromarray(laser_pattern_array)
        laser_pattern = laser_pattern.resize(img1.size, Image.LANCZOS)
        laser_pattern_array = np.array(laser_pattern)
    # Blend images: result = image1 × (1−α) + image2 × α
    blended_array = img_array * (1-alpha) + laser_pattern_array * alpha
    # Ensure output is uint8
    return np.clip(blended_array, 0, 255).astype(np.uint8)