# @Time    : 2022/3/1 15:44
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : process_pc
# @Project Name :迁移学习+tfrecord案例
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
#_______________________________________________________________________________________________________________________
#用于存储图像处理及画图的一些操作

def process_pc(image,resize_w,resize_h):
    if image.dtype!=tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    image = tf.image.resize_images(image,[resize_w,resize_h])
    return image

def visial_label(image):
    image = np.squeeze(image)
    plt.imshow(image)
    plt.show()
    plt.close()