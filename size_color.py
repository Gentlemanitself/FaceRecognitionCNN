from skimage import io,transform,color
import numpy as np
from skimage.color.colorconv import rgb2gray
from shape_label import resize_image
import cv2

def change_size_color(image_name):
    img_data=io.imread(image_name,plugin='matplotlib')
    img=color.rgb2gray(img_data)
    img=resize_image(img,64,64)
    return img

#AuImage_data=change_size_color('J:\\gt_db\\01\\01.jpg')
#cv2.imshow('image',AuImage_data)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite('J:\\', AuImage_data)  