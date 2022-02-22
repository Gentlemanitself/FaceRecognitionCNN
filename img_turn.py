from size_color import change_size_color
from skimage import io,transform,color
import numpy as np
import os
import cv2
folderList =os.listdir('J:\\gt_db')
#把存放数据文件的目录J:\\gt_db下的所有文件夹名的信息存放到一个变量folderlist中
#folderlist 是一个结构体变量数组
length=len(folderList)
AuImage_data={}
for i in range(length):
    folderName = 'J:\\gt_db\\' + folderList[i]
    AuImgList=os.listdir(folderName)
    k =len(AuImgList)
    for m in range(k):
         fileName= folderName + '\\' + AuImgList[m]
         #获取图像文件的绝对路径
         
         AuImage_data[m]=change_size_color(fileName)
         #调用函数将图片文件改成灰度图且尺寸为64*64
         #cv2.imshow('image',AuImage_data[m])
         #cv2.waitKey(0)
         #cv2.destroyAllWindows()
         rows , cols = AuImage_data[m].shape

         #得到原来图像的矩阵的参数
         MidGrayPic = np.zeros((rows , cols))
         #用得到的参数创建一个全零的矩阵，这个矩阵用来存储用下面的方法产生的灰度图像


         for p in range(rows):
             for j in range(cols):
                 #sum = 0
                 #sum = sum + AuImage_data[m][p][j]
                 #进行转化的关键公式，sum每次都因为后面的数字而不能超过255
                 MidGrayPic[p][j] = AuImage_data[m][p][j]*255
       
         str='J:\\new_data\\' +  folderList[i] + AuImgList[m][:-4] + '.jpg' 
         # 连接字符串形成生成的灰度图像的文件名，1：end-4去掉原来文件的后缀名
         cv2.imwrite(str, MidGrayPic)  
         #写文件
