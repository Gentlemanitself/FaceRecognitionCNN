import sys
import numpy as np
import os
import cv2
from PIL import Image
################################################
#读取待训练的人脸图像，指定图像路径即可
################################################
IMAGE_SIZE = 64
#将输入的图像大小统一
def resize_image(image,height = IMAGE_SIZE,width = IMAGE_SIZE):
    top,bottom,left,right = 0,0,0,0
    #获取图像大小
    h,w= image.shape
    #对于长宽不一的，取最大值
    longest_edge = max(h,w)
    #计算较短的边需要加多少像素
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    #定义填充颜色
    BLACK = [0,0,0]

    #给图像增加边界，使图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant_image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)

    return cv2.resize(constant_image,(height,width))
#读取数据
images = []     #数据集
labels = []     #标注集
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        #print(dir_item)
        full_path = path_name + '\\' + dir_item
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            #判断是人脸照片
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                #image = resize_image(image)
                #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #cv2.imwrite('./result.jpg', gray)
                images.append(image)
                labels.append(full_path)

    return images,labels

#为每一类数据赋予唯一的标签值
def label_id(label,users,user_num):
    for i in range(user_num):
        if label.endswith(users[i]):
            return i

#从指定位置读数据
def load_dataset(path_name):
    users = os.listdir(path_name)
    #print(users)
    user_num = len(users)

    images,labels = read_path(path_name)
    images_np = np.array(images)
    #每个图片夹都赋予一个固定唯一的标签
    labels_np = np.array([label_id(label,users,user_num) for label in labels])

    return images_np,labels_np

#for i in range(50):
#path_name='J:\\gt_db'

#images,labels=read_path(path_name)
#images_np,labels_np=load_dataset(path_name)
#label_np=labels_qc(labels_np)
#print(images[0].shape)
#print(len(images))
#image = cv2.imread('J:\\gt_db\\01\\01.jpg')
#fileout = 'J:\\adjust_img.jpg'
#filein = 'J:\\gt_db\\01\\01.jpg'
#img=Image.open(filein)
#(x,y) = img.size #read image size
#x_s = 250 #define standard width
#y_s = y * x_s / x #calc height based on standard width
#print(image.shape)
#out=img.resize((x_s,y_s),Image.ANTIALIAS)
#out.save(fileout)
#image = cv2.imread('fileout')
#print(image.shape)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(img.shape)
#接下来需要完成图片64*64*3到64*64的转化，以及将64*64的图片变成1*4096格式并把750图片数据合并为750*4096的矩阵