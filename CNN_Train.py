'''
使用处理后的gt_db人脸数据，利用CNN卷积神经网络实现人脸识别

'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.io as sio
import os

f = open('/test.mat','rb')
mdict = sio.loadmat(f)
# fea：数据    gnd：标签
train_data = mdict['fea']
train_label = mdict['gnd']

# 将数据分为训练数据与测试数据
train_data = np.random.permutation(train_data)
train_label = np.random.permutation(train_label)
test_data = train_data[0:64]
test_label = train_label[0:64]
np.random.seed(100)
test_data = np.random.permutation(test_data)
np.random.seed(100)
test_label = np.random.permutation(test_label)
train_data = train_data.reshape(train_data.shape[0], 64, 64 ,1).astype(np.float32) / 255

# 将标签数据改为one_hot编码格式的数据
train_labels_new = np.zeros((750, 50))
for i in range(0, 750):
    j = int(train_label[i, 0]) - 1
    train_labels_new[i, j] = 1

test_data_input = test_data.reshape(test_data.shape[0], 64, 64 ,1).astype(np.float32) / 255
test_labels_input = np.zeros((64, 50))#取64个作为测试集对于yale数据集可能大小合适，但这里用了新数据集需不需要取大一些？
for i in range(0, 64):
    j = int(test_label[i, 0]) - 1
    test_labels_input[i, j] = 1

# CNN
data_input = tf.placeholder(tf.float32, [None, 64, 64 ,1],name='data_input')

label_input = tf.placeholder(tf.float32, [None, 50])

layer1 = tf.layers.conv2d(inputs=data_input, filters=32, kernel_size=2, strides=1, padding='SAME',
                          activation=tf.nn.relu)
layer1_pool = tf.layers.max_pooling2d(layer1, pool_size=2, strides=2)
layer2 = tf.layers.conv2d(inputs=layer1_pool, filters=64, kernel_size=2, strides=1, padding='SAME',
                          activation=tf.nn.relu)
layer2_pool = tf.layers.max_pooling2d(layer2, pool_size=2, strides=2)
layer3 = tf.reshape(layer2_pool, [-1, 16 * 16 * 64])
layer3_relu1 = tf.layers.dense(layer3, 1024, tf.nn.relu)
#layer3_relu = tf.layers.dense(layer3, 1024, tf.nn.relu)
layer3_relu2= tf.layers.dense(layer3_relu1, 512, tf.nn.relu)
output = tf.layers.dense(layer3_relu2, 50)
#output = tf.layers.dense(layer3_relu, 15)
# 计算损失函数  最小化损失函数  计算测试精确度
loss = tf.losses.softmax_cross_entropy(onehot_labels=label_input, logits=output)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(label_input, axis=1), predictions=tf.argmax(output, axis=1))[1]

b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(output,b,name='logits_eval')
# 初始化 运行计算图
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(0, 25000):
        train_data_input = np.array(train_data)
        train_label_input = np.array(train_labels_new)
        sess.run([train, loss], feed_dict={data_input: train_data_input, label_input: train_label_input})
        acc = sess.run(accuracy, feed_dict={data_input: test_data_input, label_input: test_labels_input})
        print('step:%d  accuracy:%.2f%%' % (i + 1, acc * 100))
    saver.save(sess, '2590_test_model')