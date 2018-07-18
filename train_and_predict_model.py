import tensorflow as tf
import numpy as np
import glob

from PIL import Image

import utils
import skimage
import matplotlib.pyplot as plt
import os,datetime,time
from sys import getsizeof
import gzip, pickle, struct,sys
import matplotlib.pyplot as plt
import vgg16_trainable as vgg16

img_size = 224
root_dir = "/home/bwade/sunghun_taeuk/Mun/FaceTest_manyFile/"
category = os.listdir(root_dir)
category.sort()

########################################################
#
def to_category(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n= y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.int)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


########################################################
#
def train_from_pretrained():
    n_class = len(os.listdir(root_dir))
    data = np.load('./dataSet_handmade/dataSet.npz')

    x_train = data['x_train']
    y_train = data['y_train']

    y_train = to_category(y_train)

    batch_size = 64
    cost_all = []

    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, n_class])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16('./vgg16.npy', n_class = n_class)
    vgg.build(images, train_mode, int_image=True)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    print("error?")
    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    for step in range(1409110):
        batch_mask = np.random.choice(x_train.shape[0],batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]
        _, c = sess.run([train, cost], feed_dict={images: x_batch, true_out: y_batch, train_mode: True})
        if step % 1000 == 0:
            cost_all.append([step, c])
            print(step,c)

    np.savetxt("cost.txt", cost_all)
    # test classification again, should have a higher probability about tiger

    # test accuracy 계산
    n_sample = 50
    n = int(x_train.shape[0] / n_sample)
    accuracy = 0
    for i in range(n):
        prob = sess.run(vgg.prob, feed_dict={images: x_train[i*50:(i+1)*50], train_mode: False})
        prob = np.argmax(prob, axis=1)
        y_batch = y_train[i*50:(i+1)*50]
        accuracy += np.sum(prob == y_batch) / float(n_sample)
    print("accuracy: ", accuracy/n)

    # test save
    vgg.save_npy(sess, './test-save.npy')

########################################################
#
def predict_from_pretrained():
    n_class = len(os.listdir(root_dir))
    
    test_source = 1
    if test_source == 1:
        filenames = ['./test_data/1.jpg', './test_data/2.jpg', './test_data/3.jpg', './test_data/4.jpg', './test_data/5.jpg', './test_data/6.jpg','./test_data/7.jpg', './test_data/8.jpg', 
		'./test_data/9.jpg', './test_data/10.jpg', './test_data/11.jpg', './test_data/12.jpg', './test_data/13.jpg', './test_data/14.jpg', './test_data/15.jpg', './test_data/16.jpg', './test_data/17.jpg',
		'./test_data/18.jpg', './test_data/19.jpg', './test_data/20.jpg', './test_data/21.jpg', './test_data/22.jpg', './test_data/23.jpg']
        label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        
        test_batch = []
        for fname  in filenames:
            img = Image.open(fname)
            img = img.convert("RGB")
            img = img.resize((img_size, img_size))
            data = np.asarray(img)
            test_batch.append(data)

        test_batch = np.array(test_batch)

    elif test_source == 2:
        data = np.load('./dataSet/dataSet_shuffle_0.npz')

        test_batch = data['x_train']
        label = data['y_train']

        ndata = min(50, test_batch.shape[0])
        batch_mask = np.random.choice(test_batch.shape[0], ndata)

        test_batch = test_batch[batch_mask]
        label = label[batch_mask]

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        train_mode = tf.placeholder(tf.bool)
        vgg = vgg16.Vgg16('./test-save.npy', trainable = False, n_class = n_class)

        with tf.name_scope("content_vgg"):
            vgg.build(images, train_mode, int_image=True)

        prob = sess.run(vgg.prob, feed_dict = {train_mode: False, images: test_batch})
        #print(prob)
        prob_argmax = np.argmax(prob,axis=1)
        for i in range(min(100, test_batch.shape[0])):
            print(i, "who: ", category[label[i]], "\t\t | predict: ", category[prob_argmax[i]],
                  "\t\t | result: ", label[i] == prob_argmax[i])

        print("acc: ", np.sum(label==prob_argmax)/float(test_batch.shape[0]))

if __name__ == "__main__":
    s = time.time()

    train_from_pretrained()
    # predict_from_pretrained()

    e=time.time()

    print("경과시간: ", e-s, "sec")
