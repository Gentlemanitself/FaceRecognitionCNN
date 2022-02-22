from tkinter import filedialog
from PIL import Image, ImageTk
from pylab import *
import tkinter as tk
import numpy as np
import datetime
from skimage import io, transform, color
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def button1():
    global file1
    file1 = tk.filedialog.askopenfilename()
    txt_path1.set(file1)


def read_one_image(path):
    rgb = io.imread(path)  # 读取图片
    gray = color.rgb2gray(rgb)  # 将彩色图片转换为灰度图片
    dst = transform.resize(gray, (64, 64))
    image_arr = np.array(dst)
    result = image_arr.reshape(64, 64, 1).astype(np.float32)
    return result


def button2():
    # fd=txt_path1.set(file1)
    #file = r'file1'
    t1 = datetime.datetime.now()

    with tf.Session() as sess:
        data = []
        data1 = read_one_image(file1)

        data.append(data1)

        saver = tf.train.import_meta_graph('./TrainMod/2590_test_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./TrainMod'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("data_input:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess.run(logits, feed_dict)

        txt_path2.set(tf.argmax(classification_result, 1).eval())

        t2 = datetime.datetime.now()
    t = t2-t1
    txt_path3.set(t)


def resize(w, h, w_box, h_box, im):
    f1 = 1.0*w_box/w
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    width = int(w*factor)
    height = int(h*factor)
    return im.resize((width, height), Image.ANTIALIAS)


def Diff():
    #ims = array(Image.open(file1))
    # imr=ims.reshape(1,784)
    # print(imr.shape)
    im = Image.open(file1)
    w, h = im.size
    global img
    imr = resize(w, h, w_box, h_box, im)
    img = ImageTk.PhotoImage(imr)
    label_img = tk.Label(window, image=img, width=w_box, height=h_box)
    label_img.pack(padx=5, pady=5)
    label_img.place(x=120, y=200)
    # imshow(im)
    # show()
    #photo = tk.PhotoImage(file1)
    #imglabel = tk.Label(window,image=photo)
    # label_im.pack()
    # imglabel.pack(side=tk.LEFT)


w_box = 400
h_box = 400
window = tk.Tk()
window.title('人脸识别系统')
window.geometry('1200x600')

label0 = tk.Label(window, text='选择要识别的图片：', fg='blue',
                  font=('Microsoft YaHei UI', 18)).place(x=60, y=30)

label1 = tk.Label(window, text='图片路径：', font=('Microsoft JHengHei UI', 11)).place(x=30, y=80)
label2 = tk.Label(window, text='识别结果：', font=('Microsoft JHengHei UI', 11)).place(x=30, y=110)
label3 = tk.Label(window, text='识别用时：', font=('Microsoft JHengHei UI', 11)).place(x=550, y=110)


txt_path1 = tk.StringVar()
text1 = tk.Entry(window, textvariable=txt_path1, show=None, width=60)
txt_path2 = tk.StringVar()
text2 = tk.Entry(window, textvariable=txt_path2, show=None, width=60)
txt_path3 = tk.StringVar()
text3 = tk.Entry(window, textvariable=txt_path3, show=None, width=60)
text1.place(x=120, y=80)
text2.place(x=120, y=110)
text3.place(x=650, y=110)
button1 = tk.Button(window, width=8, height=1, text='选择图片',font=('Microsoft JHengHei UI', 10),
                    bg='#C0C0C0', command=button1).place(x=550, y=80)
t1 = datetime.datetime.now()
button2 = tk.Button(window, width=20, height=1, text='识别图片',font=('Microsoft JHengHei UI', 10),
                    fg='black', bg='#C0C0C0', command=button2).place(x=350, y=150)
t2 = datetime.datetime.now()
button3 = tk.Button(window, width=20, height=1, text='显示图片',font=('Microsoft JHengHei UI', 10),
                    fg='black', bg='#C0C0C0', command=Diff).place(x=150, y=150)


window.mainloop()
