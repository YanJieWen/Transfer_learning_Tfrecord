# @Time    : 2022/3/3 14:03
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : demo
# @Project Name :迁移学习+tfrecord案例
import random
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
from utilss.match_math import *
from utilss.process_pc import *
def demo(pc_file,h,w,meta_file,model_fac,all_label):
    image_raw_data = gfile.FastGFile(pc_file,'rb').read()#jpeg格式读图
    image = tf.image.decode_jpeg(image_raw_data)#将jpeg格式解码
    image = process_pc(image,h,w)#将图片处理为适合网络输入的尺寸
    image = tf.expand_dims(image,0)
    saver = tf.train.import_meta_graph(meta_file)  # 加载计算图
    model_file = tf.train.latest_checkpoint(model_fac).replace('\\', '/')  # 获取最新的ckpt权重文件
    graph = tf.get_default_graph()
    oout = graph.get_tensor_by_name('InceptionV3/Predictions/Softmax:0')  # 根据节点名称，无需重复加载计算图

    with tf.Session() as sess:
        saver.restore(sess, model_file)
        image_value = sess.run(image)
        pred = sess.run(oout, feed_dict={'Placeholder:0': image_value})  # 通过变量
        print(pred)
        index_pred = np.argmax(pred)
        return all_label[index_pred],image_value

