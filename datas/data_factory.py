# @Time    : 2022/3/1 9:59
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : data_factory
# @Project Name :迁移学习+tfrecord案例
#_______________________________________________________________________________________________________________________
import glob#查找文件路径名
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from utilss.match_math import *
from utilss.process_pc import *
#_______________________________________________________________________________________________________________________
#定义一个类，用于将图片数据保存为Tensorflow专用的数据格式Tfrecords
class To_TfRecord():
    def __init__(self,input_data,train_pet,valid_pet,num_shard,all_label):
        self.input_data = input_data
        self.train_pet = train_pet
        self.valid_pet = valid_pet
        self.num_shard = num_shard
        self.all_label = all_label
    def scan_dir(self):#自上而下扫描目录，将各个子目录名称以列表形式存储
        return [x[0] for x in os.walk(self.input_data)]
    def get_pc_name(self):#通过glob函数正则化匹配位于每个目录下存储的图片路径
        file_list = []
        sub_dirs = self.scan_dir()
        for sub_dir in sub_dirs:
            extensions = ['jpg', 'jpeg']
            dir_name = os.path.basename(sub_dir)
            for extension in extensions:
                file_glob = os.path.join(self.input_data, dir_name, '*.' + extension).replace('\\', '/')
                file_glob_ = [name.replace('\\', '/') for name in glob.glob(file_glob)]
                if len(file_glob_):
                    file_list.append(file_glob_)
        return file_list
    def make_data(self):#返回列表[train_file,valid_file,test_file]
        # 打乱数据集
        file_list = self.get_pc_name()
        file_name = [name for names in file_list for name in names]
        np.random.shuffle(file_name)
        train_file = file_name[:int(self.train_pet* len(file_name))]
        valid_file = file_name[int(self.train_pet * len(file_name)):
                               int((self.train_pet+self.valid_pet) * len(file_name))]
        test_file = file_name[int((self.train_pet+self.valid_pet) * len(file_name)):]
        all_data = [train_file, valid_file, test_file]
        return all_data
    def get_tfrecord(self):#获取分别为train，test和validation的tfrecord,单个样例存储模式
        all_data = self.make_data()
        sess = tf.Session()
        for i in range(self.num_shard):
            filename = './datasets/data.tfrecords-%.5d-of-%.5d' % (i, self.num_shard)
            writer = tf.python_io.TFRecordWriter(filename)
            for single_file_name in all_data[i]:
                image_raw_data = gfile.FastGFile(single_file_name, 'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                shape_0 = tf.shape(image)
                image_value = sess.run(image)
                shape_value = sess.run(shape_0)
                image_raw = image_to_str(image_value)  # 图像转为字符串
                label = re_match(single_file_name)  # 标签
                label_raw = transfer_label(label,self.all_label)#将字符串数据转为整型用于网络输入
                # 对字符串型进行编码
                example = tf.train.Example(features=tf.train.Features(feature={'label': _int64_feature(label_raw),
                                                                               'image_raw': _bytes_feature(image_raw),
                                                                               'w': _int64_feature(shape_value[0]),
                                                                               'h':_int64_feature(shape_value[1]),
                                                                               'c':_int64_feature(shape_value[2])}))
                writer.write(example.SerializeToString())
            writer.close()

def _int64_feature(value):#整型
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):#字符型
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):#浮点型
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def image_to_str(image):#将图像矩阵转化为一个字符串
    return image.tostring()