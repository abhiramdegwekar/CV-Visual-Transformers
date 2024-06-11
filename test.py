import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def get_pixel(img, center, x, y): 
    new_value = 0
    try: 
        # If local neighbourhood pixel value is greater than or equal to center pixel values then set it to 1 
        if img[x][y] >= center: 
            new_value = 1
    except:  
        pass
    return new_value 
   
# Function for calculating LBP 
def lbp_calculated_pixel(img, x, y): 
    center = img[x][y] 
    val_ar = [] 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
    val_ar.append(get_pixel(img, center, x-1, y)) 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
    val_ar.append(get_pixel(img, center, x, y-1)) 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
    val = 0
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
    return val 



# path = 'fer2013/train/angry/Training_3908.jpg'
# img_bgr = cv2.imread(path, 1) 
# height, width, _ = img_bgr.shape 
# img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 
   
# # Create a numpy array as the same height and width of RGB image 
# img_lbp = np.zeros((height, width), np.uint8) 
   
# for i in range(0, height): 
#     for j in range(0, width): 
#         img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j) 


class LBPExtractor(nn.Module):
    def __init__(self):
        super(LBPExtractor, self).__init__()

    def forward(self, rgb_image):
        height, width = rgb_image.shape 
        lbp_feature_image = np.zeros((height, width), np.uint8) 

        for i in range(0, height): 
            for j in range(0, width): 
                lbp_feature_image[i, j] = lbp_calculated_pixel(rgb_image, i, j) 

        return lbp_feature_image

class FeatureConcatenator(nn.Module):
    def __init__(self):
        super(FeatureConcatenator, self).__init__()

    def forward(self, rgb_image, lbp_feature_image):
        if not isinstance(lbp_feature_image, torch.Tensor):
            lbp_feature_image = torch.from_numpy(lbp_feature_image)
            rgb_image =  torch.from_numpy(rgb_image)
        # Concatenate the RGB image and LBP feature image along the channel dimension
        concatenated_features = torch.cat((rgb_image, lbp_feature_image), dim=1)
        return concatenated_features

# Load pre-trained ResNet18
resnet_rgb = models.resnet18(weights=True)
resnet_lbp = models.resnet18(weights=True)

# Modify ResNet18 to output features up to stage 5
modules_rgb = list(resnet_rgb.children())[:-2]  # Remove last 2 layers (avgpool and fc)
resnet_rgb = nn.Sequential(*modules_rgb)

modules_lbp = list(resnet_lbp.children())[:-2]  # Remove last 2 layers (avgpool and fc)
resnet_lbp = nn.Sequential(*modules_lbp)

# Example usage:
path = 'fer2013/train/angry/Training_3908.jpg'
img_bgr = cv2.imread(path, 1) 
height, width, _ = img_bgr.shape 
rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

lbp_extractor = LBPExtractor()
lbp_feature_image = lbp_extractor(rgb_image)

concatenator = FeatureConcatenator()


# Reshape LBP image to have 3 channels
lbp_image_3channel = np.repeat(lbp_feature_image[..., None], 3, axis=2)

# Concatenate BGR and reshaped LBP images
concatenated_image = np.concatenate((img_bgr, lbp_image_3channel), axis=2)


img_bgr_tensor = torch.from_numpy(img_bgr).unsqueeze(0).unsqueeze(0)  # Assuming batch size is 1 and channels is 1
rgb_features = resnet_rgb(img_bgr_tensor)