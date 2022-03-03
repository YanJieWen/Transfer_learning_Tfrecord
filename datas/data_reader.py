# @Time    : 2022/3/2 14:10
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : data_reader
# @Project Name :迁移学习+tfrecord案例
#_______________________________________________________________________________________________________________________
import tensorflow as tf
import numpy as np
from utilss.process_pc import *
#_______________________________________________________________________________________________________________________
class Read_TfRecord():
    def __init__(self,data_type,input_h,input_w,batch_size):
        self.data_type = data_type
        self.input_h = input_h
        self.input_w = input_w
        self.batch_size = batch_size
    def get_features(self):#获取一个解析样本
        files = tf.train.match_filenames_once('./datasets/data.tfrecords-*{}-*'.format(self.data_type))#使用该函数前要先初始化才能run
        filename_queue = tf.train.string_input_producer(files, shuffle=False)
        reader = tf.TFRecordReader()  # 读取并解析一个样本
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64),
                                                                         'image_raw': tf.FixedLenFeature([], tf.string),
                                                                         'w': tf.FixedLenFeature([], tf.int64),
                                                                         'h': tf.FixedLenFeature([], tf.int64),
                                                                         'c': tf.FixedLenFeature([], tf.int64)})
        return features
    def parse_data(self):
        features = self.get_features()
        decode_image = tf.decode_raw(features['image_raw'], tf.uint8)#转换数据的类型一定要和存储数据前的类型一样否则会导致解析长度不等，专用于解析字符串类型数据
        image_change_shape = [features['w'], features['h'], features['c']]
        shape_ = tf.convert_to_tensor(image_change_shape)  # 将list转化为tensor，用于tf.reshape动态改变张量形状
        new_image = tf.reshape(decode_image, shape_)
        image = process_pc(new_image, self.input_w, self.input_h)#读出一个样例来之后再进行处理
        decode_label = tf.cast(features['label'], tf.int64)
        image.set_shape([self.input_w, self.input_h, 3])#set_shape通过指定静态形状满足tf.train.shuffle_batch的要求
        return image,decode_label
    def gen_batch(self):#整理为batch，以tensor形式输出
        image,decode_label = self.parse_data()
        capacity = 1000 + 3 * self.batch_size  # 固定公式
        image_batch, label_batch = tf.train.shuffle_batch([image, decode_label], batch_size=self.batch_size,
                                                          capacity=capacity, min_after_dequeue=1000)
        return image_batch,label_batch#返回一个tensor包括image_batch和label_batch,生成一个类似placehordler