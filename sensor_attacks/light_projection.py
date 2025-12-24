"""
Create Light Projection Attack Module
This module implements a light projection attack by overlaying a watermark on an image.
"""
import numpy as np
from PIL import Image

def light_projection(origin_array, watermark_path, transparency=0.6, x=0, y=0):
    """
    Apply light projection attack by overlaying a watermark on the original image.
    
    Args:
        watermark_path: Path to the watermark image to be projected
        origin_array: Numpy array of the original image
        transparency: Float value between 0 and 1 for watermark transparency
        x: Horizontal offset for watermark position (0 means center)
        y: Vertical offset for watermark position (0 means center)
        
    Returns:
        Numpy array with the watermark projected onto the original image
    """
    # Load watermark image from path
    watermark = Image.open(watermark_path)
    # Convert numpy array to PIL Image for origin
    origin = Image.fromarray(origin_array)
    
    # Convert both images
    watermark = watermark.convert('RGBA')
    origin = origin.convert('RGBA')
            
    
    # Adjust the transparency
    datas = watermark.getdata()
    newData = []
    for item in datas:
        newData.append((item[0], item[1], item[2], int(item[3] * transparency)))
    watermark.putdata(newData)
    
    # Create base image and paste
    baseImg = Image.new('RGBA', origin.size)
    baseImg.paste(origin, (0, 0), origin)
    # Calculate position with offset (x, y). 0,0 means center
    center_x = int((origin.width - watermark.width) / 2)
    center_y = int((origin.height - watermark.height) / 2)
    position = (center_x + x, center_y + y)
    baseImg.paste(watermark, position, watermark)
    
    # Convert the final image back to 'RGB' mode and return as numpy array
    result_img = baseImg.convert('RGB')
    return np.array(result_img)
