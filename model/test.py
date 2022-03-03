# @Time    : 2022/3/3 10:04
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : test
# @Project Name :迁移学习+tfrecord案例
import tensorflow as tf
from datas.data_reader import *
import tensorflow as tf
class Test_model():
    def __init__(self,meta_file,model_fac,h,w,batch_size):
        self.meta_file = meta_file
        self.model_fac = model_fac
        self.h = h
        self.w = w
        self.batch_size = batch_size
    def test_setting(self):
        saver = tf.train.import_meta_graph(self.meta_file)#加载计算图
        model_file = tf.train.latest_checkpoint(self.model_fac).replace('\\','/')#获取最新的ckpt权重文件
        graph = tf.get_default_graph()
        oout = graph.get_tensor_by_name('InceptionV3/Predictions/Softmax:0')#根据节点名称，无需重复加载计算图
        test_sess = tf.Session()
        test_reader = Read_TfRecord(2, self.h, self.w, self.batch_size)
        test_exam, test_label = test_reader.gen_batch()
        test_sess.run((tf.global_variables_initializer(),tf.local_variables_initializer()))
        #先不放入for循环中产生多个batch,可以定义较大的batch_size进行测试
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=test_sess, coord=coord)
        saver.restore(test_sess,model_file)
        cur_test_image, cur_test_label = test_sess.run([test_exam, test_label])
        pred =  test_sess.run(oout,feed_dict={'Placeholder:0':cur_test_image})#通过变量
        coord.request_stop()
        coord.join(threads)
        return pred,cur_test_label


