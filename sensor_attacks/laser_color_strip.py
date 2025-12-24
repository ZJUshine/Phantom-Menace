"""
Laser Color Strip Attack Module
This module implements a laser color strip attack on images.
"""
import numpy as np
import math

def color_func(p, q, start_row, strength, h, w):
    """
    Calculate color function value based on Gaussian distribution.
    
    Args:
        p: Row position
        q: Column position
        start_row: Starting row for the effect
        strength: Intensity of the effect
        h: Image height
        w: Image width
        
    Returns:
        Float value representing color intensity
    """
    # Gaussian distribution
    rho = 0.0
    part_a = ((p-start_row-h/2)**2)/((h/2)**2)
    part_b = ((q-w/2)**2)/((w/4)**2)
    part_c = 2*(p-start_row-h/2)*(q-w/2)/(h*w/8)*rho
    f_value = 1/(2*math.pi*h/2*w/4*np.sqrt(1-rho**2))*np.exp(-1/(2*(1-rho**2))*(part_a+part_b+part_c))*strength*h*w
    return f_value

def laser_color_strip(original_image_array, red_percent=0.6, green_percent=0.1, blue_percent=0.1, strength=1500):
    """
    Apply laser color strip attack to an image.
    
    Args:
        original_image_array: Numpy array of the image to be attacked
        red_percent: Percentage of red color to add
        green_percent: Percentage of green color to add
        blue_percent: Percentage of blue color to add
        strength: Intensity of the laser effect
        
    Returns:
        Numpy array with laser color strip attack applied
    """
    # Create a copy of the input array to avoid modifying the original
    result_array = np.copy(original_image_array)
    
    # Get image dimensions
    height, width = original_image_array.shape[:2]
    
    # Apply color strip in the middle third of the image
    start_row = height//3
    end_row = height//3*2
    
    for i in range(0, width):
        for j in range(start_row, end_row):
            # Calculate intensity based on position
            intensity = color_func(p=j, q=i, start_row=start_row, strength=strength, h=height, w=width)
            
            # Calculate color additions
            add_red = intensity * red_percent
            add_green = intensity * green_percent
            add_blue = intensity * blue_percent
            
            # Get current pixel values
            current_pixel = result_array[j, i].copy()
            
            # Add intensity values
            current_pixel[0] = min(255, current_pixel[0] + add_red)
            current_pixel[1] = min(255, current_pixel[1] + add_green)
            current_pixel[2] = min(255, current_pixel[2] + add_blue)
            
            # Update pixel in result array (ensure integer values)
            result_array[j, i] = np.array(current_pixel, dtype=np.uint8)
    
    return result_array