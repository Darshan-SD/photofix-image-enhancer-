import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = './photos/3.jpeg'  
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def correct_brightness(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    corrected_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return corrected_image

corrected_image = correct_brightness(image)

corrected_image_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.imshow(corrected_image_rgb)
plt.title('Corrected Image')
plt.axis('off')
plt.show()
