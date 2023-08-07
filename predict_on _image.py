import os
import cv2
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
#import PIL
from tensorflow.keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

#from tensorflow.keras.utils import to_categorical, Sequence

#from sklearn.model_selection import train_test_split

from simple_multi_unet_model import multi_unet_model, jacard_coef  

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
import segmentation_models as sm

sm.set_framework('tf.keras')
sm.framework()

from smooth_tiled_predictions import predict_img_with_smooth_windowing


def label_to_rgb(predicted_image):
    
    Forest = '#3C1098'.lstrip('#')
    Forest = np.array(tuple(int(Forest[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    
    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    Road = '#6EC1E4'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    
    Vegetation =  'FEDD3A'.lstrip('#') 
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    
    Water = 'E2A929'.lstrip('#') 
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    
    Unlabeled = '#9B9B9B'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
    
    
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
    segmented_img[(predicted_image == 0)] = Forest
    #segmented_img[(predicted_image == 1)] = Land
    #segmented_img[(predicted_image == 2)] = Road
    #segmented_img[(predicted_image == 3)] = Vegetation
    #segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)

#changed to import numpy array directly

#img = cv2.imread('images/waikato_crop.png', 1)
img = np.load('images/image_numpy.npy')
#original_mask = cv2.imread("test_data/mask_part_008.png", 1)
#original_mask = cv2.cvtColor(original_mask,cv2.COLOR_BGR2RGB)
print('image loaded')
#weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
#dice_loss = sm.losses.DiceLoss(class_weights=weights) 
#focal_loss = sm.losses.CategoricalFocalLoss()
#total_loss = dice_loss + (1 * focal_loss)
#model = tf.keras.models.load_model('./models/', custom_objects={'dice_loss_plus_2focal_loss': total_loss, 'jacard_coef':jacard_coef})
model = tf.keras.models.load_model('./models/', custom_objects={'jacard_coef':jacard_coef})

# size of patches
patch_size = 256

# Number of classes 
n_classes = 2

#Unsmoothed patches try due to memory over error with smoothed patches

#note this now works with 4 bands for a 4 band model. Imports a numpy array directly

SIZE_X = (img.shape[2]//patch_size)*patch_size #Nearest size divisible by our patch size
SIZE_Y = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
bands = img.shape[0]
#large_img = PIL.Image.fromarray(img)
image = img[:,:SIZE_Y,:SIZE_X]  #Crop from top left corner
image = np.moveaxis(image, 0, 2) #move the bands to the end of the array for patchify      

print('patchifying now...')
patches_img = patchify(image, (patch_size, patch_size, bands), step=patch_size)  #Step=256 for 256 patches means no overlap
patches_img = patches_img[:,:,0,:,:,:]

print('patching done, onto predicting...')
patched_prediction = []
for i in range(patches_img.shape[0]):
    
    for j in range(patches_img.shape[1]):
        print('\rpredicting image:', i, j, end='')
        single_patch_img = patches_img[i,j,:,:,:]
        
        #Use minmaxscaler instead of just dividing by 255. 
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0, :,:]
                                 
        patched_prediction.append(pred)

print('prediction done, unpatchifying')
patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1], 
                                            patches_img.shape[2], patches_img.shape[3]])

unpatched_prediction = unpatchify(patched_prediction, (image.shape[0], image.shape[1]))

prediction_with_smooth_blending=label_to_rgb(unpatched_prediction)

print('saving image')
cv2.imwrite('forest_predicted.png', prediction_with_smooth_blending)
#output image is smaller than input because of patches. Alignment with top left corner works out to overlap nicely
