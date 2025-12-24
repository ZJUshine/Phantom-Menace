"""
Ultrasound Blur Attack Module
This module implements an ultrasound blur attack on images.
"""
import numpy as np
import math

def ultrasound_blur(img_array, theta=0, dx=10, dy=10, S=0, max_samples=100):
    """
    Apply ultrasound blur attack to an image.
    
    Args:
        img_array: Numpy array of the image to be attacked
        theta: Rotation angle in degrees
        dx: X-axis displacement
        dy: Y-axis displacement
        S: Scale factor
        max_samples: Maximum number of samples to take for each pixel
        
    Returns:
        Numpy array with ultrasound blur attack applied
    """
    imgheight, imgwidth = img_array.shape[:2]
    imgarray = img_array.astype(np.float32)  # 使用float32避免整数溢出
    c0 = int(imgheight / 2)
    c1 = int(imgwidth / 2)
    delta = np.arctan(dy/dx)
    L = np.sqrt(dx*dx+dy*dy)
    theta = theta / 180 * math.pi
    blurred_imgarray = np.copy(imgarray)
    
    # 为了提高性能，只处理图像的一部分区域
    # 可以根据需要调整采样率
    sample_rate = 2  # 每隔2个像素处理一个
    
    for x in range(0, imgheight, sample_rate):
        for y in range(0, imgwidth, sample_rate):
            R = math.sqrt((x - c0) ** 2 + (y - c1) ** 2)
            alpha = math.atan2(y - c1, x - c0)
            X_cos = L * math.cos(delta) - S * R * math.cos(alpha)
            Y_sin = L * math.sin(delta) - S * R * math.sin(alpha)
            N = int(max(abs(R * math.cos(alpha + theta) + X_cos + c0 - x),
                        abs(R * math.sin(alpha + theta) + Y_sin + c1 - y)))
            
            # 限制最大采样点数，避免过度计算
            N = min(N, max_samples)
            
            if N <= 0:
                continue
                
            count = 0
            sum_r, sum_g, sum_b = 0.0, 0.0, 0.0  # 使用浮点数避免溢出
            
            for i in range(0, N + 1):
                n = i / N
                xt = int(R * math.cos(alpha + n * theta) + n * X_cos + c0)
                yt = int(R * math.sin(alpha + n * theta) + n * Y_sin + c1)
                
                if xt < 0 or xt >= imgheight:
                    continue
                elif yt < 0 or yt >= imgwidth:
                    continue
                else:
                    sum_r += float(imgarray[xt, yt][0])
                    sum_g += float(imgarray[xt, yt][1])
                    sum_b += float(imgarray[xt, yt][2])
                    count += 1
                    
            if count > 0:
                # 计算平均值并应用到当前像素及其周围像素
                avg_r = sum_r / count
                avg_g = sum_g / count
                avg_b = sum_b / count
                
                # 填充采样区域内的所有像素
                for dx in range(sample_rate):
                    for dy in range(sample_rate):
                        nx, ny = x + dx, y + dy
                        if nx < imgheight and ny < imgwidth:
                            blurred_imgarray[nx, ny] = np.array([avg_r, avg_g, avg_b])
    
    # 确保值在有效范围内
    blurred_imgarray = np.clip(blurred_imgarray, 0, 255)
    return blurred_imgarray.astype(np.uint8)