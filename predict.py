import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

import os,datetime,time
import vgg16_trainable as vgg16

img_size = 224
root_dir = "F:\Data\FaceTest_manyFile"
category = os.listdir(root_dir)

########################################################
def predict_from_pretrained(image):
    # n_class = len(os.listdir(root_dir))
    n_class = 991

    test_batch = []
    print(image.shape)
    print(1)

    img = resize_with_pad(image)
    print(img.shape)
    print(2)

    img = img.reshape((1, 224, 224, 3))
    print(img.shape)
    print(3)

    data = np.asarray(img)
    print(data.shape)
    print(3)

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        train_mode = tf.placeholder(tf.bool)
        vgg = vgg16.Vgg16('./test-save.npy', trainable = False, n_class = n_class)

        with tf.name_scope("content_vgg"):
            vgg.build(images, train_mode, int_image=True)

        prob = sess.run(vgg.prob, feed_dict = {train_mode: False, images: data})
        #print(prob)
        prob_argmax = np.argmax(prob, axis=1)
        print("predict: ", category[prob_argmax[0]])

        return category[prob_argmax[0]]

def resize_with_pad(image, height=224, width=224):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
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
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image

