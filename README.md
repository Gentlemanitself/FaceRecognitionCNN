# FaceRecognitionCNN

 A FaceRecognition network using CNN with GUI

## 仓库文件结构

- TrainMod 目录下是保存的数据文件
- CNN_Train.py CNN卷积网络搭建及训练文件
- img_turn.py 批量处理原始图像（调用shape_label的resize_image，size_color的change_size_color）
- img_turn_to_mat.m 制作test.mat人脸数据集
- Tk_Gui.py GUI接口

## 程序架构介绍

![](/ForREADME/structure.jpg)

### 训练模型

由于训练集数目较大而神经网络较为简单，固对训练的次数与精度提出了较高的要求。且此模型在test.mat下在25000次训练集下准确率对于次数依然较为敏感。

![](/ForREADME/train.jpg)

NVIDIA的CUDA模块具有用于加速 AI 的全速混合精度 Tensor Core 核心。计算中实测Capability=7.5的RTX计算卡较于CoffeeLake的Intel CPU 4.0GHz可节约90%的时间。但由于Tensorflow-GPU的环境搭建较为复杂，为方便精确调参使用SSH连接远程主机，在Linux服务器环境下进行训练。

### 保存模型

TensorFlow通过tf.train.Saver类实现神经网络模型的保存和提取。tf.train.Saver对象saver的save方法将TensorFlow模型保存到指定路径中
checkpoint文件会记录保存信息，通过它可以定位最新保存的模型，保存了一个录下多有的模型文件列表：
.meta文件保存了当前保存了TensorFlow计算图的结构信息
.index文件保存了当前参数名
.data文件保存了当前每个变量的取值，此处文件名的写入方式会因不同参数的设置而不同，但加载restore时的文件路径名是以checkpoint文件中的“model_checkpoint_path”值决定的。

### 加载模型

当我们基于checkpoint文件(ckpt)加载参数时，实际上我们使用Saver.restore取代了initializer的初始化

加载这个已保存的TensorFlow模型的方法是`saver.restore(sess,"./Model/model.ckpt")`，加载模型的代码中也要定义TensorFlow计算图上的所有运算并声明一个`tf.train.Saver`类，不同的是加载模型时不需要进行变量的初始化，而是将变量的取值通过保存的模型加载进来，注意加载路径的写法。若不希望重复定义计算图上的运算，可直接加载已经持久化的图，`saver =tf.train.import_meta_graph("Model/model.ckpt.meta")`。

`tf.train.Saver`类也支持在保存和加载时给变量重命名，声明Saver类对象的时候使用一个字典dict重命名变量即可，{"已保存的变量的名称name": 重命名变量名}，`saver = tf.train.Saver({"v1":u1, "v2": u2})`即原来名称name为v1的变量现在加载到变量u1（名称name为other-v1）中。

上一条做的目的之一就是方便使用变量的滑动平均值。如果在加载模型时直接将影子变量映射到变量自身，则在使用训练好的模型时就不需要再调用函数来获取变量的滑动平均值了。

本次实验中恢复了整张神经网络的参数和结构以便计算

```python
saver = tf.train.import_meta_graph('./TrainMod/2590_test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./TrainMod'))
graph = tf.get_default_graph()
```

## 使用Tkinter实现交互的窗口

Tkinter模块（“Tk接口”）是Tk GUI工具包的标准Python接口。主要使用一下几个部分。

1. `mainloop`函数，这个函数将让窗口等待用户与之交互，直到我们关闭它。
2. `Tkinter. Label`标签控件指定的窗口中显示的文本和图像。标签控件（Label）指定的窗口中显示的文本和图像。
3. `Tkinter. StringVar`在使用界面编程的时候，有些时候是需要跟踪变量的值的变化，以保证值的变更随时可以显示在界面上。由于python无法做到这一点，所以使用了tcl的相应的对象，也就是`StringVar、BooleanVar、DoubleVar、IntVar`所需要起到的作用
4. `Tkinter.button` 按钮组件用于在 Python 应用程序中添加按钮，按钮上可以放上文本或图像，按钮可用于监听用户行为，能够与一个 Python 函数关联，当按钮被按下时，自动调用该函数。
