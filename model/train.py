# @Time    : 2022/3/2 16:39
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : train
# @Project Name :迁移学习+tfrecord案例
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
from datas.data_reader import *
class Train_model():
    def __init__(self,trian_file,ckpt_file,lr,epoch,iter,n_classes,log_dir,w,h,batch_size):
        self.ckpt_file = ckpt_file
        self.lr=lr
        self.epoch=epoch
        self.iter = iter
        self.n_classes = n_classes
        self.log_dir = log_dir
        self.w = w
        self.h = h
        self.batch_size = batch_size
        self.train_file = trian_file
    def place_gen(self):#在此处定义的目的是
        images = tf.placeholder(tf.float32, [None, self.w, self.h, 3])
        labels = tf.placeholder(tf.int64, [None], name='labels')
        return images,labels
    def define_variable(self):
        images, labels = self.place_gen()
        # 定义参数模型
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, _ = inception_v3.inception_v3(images, num_classes=self.n_classes)  # 输出结果，加载模型
        # 获取需要训练的变量
        ckpt_exclue_scopes = 'InceptionV3/Logits,InceptionV3/AuxLogits'  # 参数的前缀
        trainable_scopes = 'InceptionV3/Logits,InceptionV3/AuxLogits'
        # 获取需要训练变量的列表
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]
        variable_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variable_to_train.extend(variables)
        # 获取所有需要从谷歌训练好的模型中加载的参数
        exclusions = [scope.strip() for scope in ckpt_exclue_scopes.split(',')]
        variables_to_restore = []
        for var in slim.get_model_variables():
            # var.name是tensor的名字有：0，而var.op.name是操作的名字
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):  # 此句是删除以该层开始下所有变量的节点,op.name是操作的名字，name是tensor的名字，op.name为tensor：0
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        tf.losses.softmax_cross_entropy(tf.one_hot(labels, self.n_classes), logits, weights=1.0)
        train_step = tf.train.RMSPropOptimizer(self.lr).minimize(tf.losses.get_total_loss())
        with tf.name_scope('eval'):
            correct_pr = tf.equal(tf.argmax(logits, 1), labels)
            eval_step = tf.reduce_mean(tf.cast(correct_pr, tf.float32))  # 将boolen类型转化为实数类型
            tf.summary.scalar('eval_correct', eval_step)
        load_fn = slim.assign_from_checkpoint_fn(self.ckpt_file, variables_to_restore, ignore_missing_vars=True)#加载的模型权重，选择性恢复权值
        return train_step,load_fn,images,labels,eval_step
    def train_setting(self):
        train_step,load_fn,images,labels,eval_step= self.define_variable()
        merged = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=3)#保存模型权重
        train_reader = Read_TfRecord(0, self.h, self.w, self.batch_size)  # 文件0是训练集，1是验证集，2是测试集
        validat_reader = Read_TfRecord(1, self.h, self.w, self.batch_size)  # 验证集的batch_size可以大一些
        train_exam, train_label = train_reader.gen_batch()
        valid_exam, valid_label = validat_reader.gen_batch()
        with tf.Session() as sess:
            writer =tf.summary.FileWriter(self.log_dir,sess.graph)
            sess.run((tf.global_variables_initializer(),tf.local_variables_initializer())) #由于match_filenames_once的存在需要同时初始化全局变量和局部变量
            load_fn(sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            for i in range(self.epoch):
                for j in range(self.iter):
                    cur_train_image,cur_train_label = sess.run([train_exam,train_label])
                    _,summary = sess.run([train_step,merged],feed_dict={images:cur_train_image,labels:cur_train_label})
                    writer.add_summary(summary,i*self.epoch+j)
                if i%2==0 or i+1 == self.epoch:
                    saver.save(sess,self.train_file,global_step=i)
                    cur_valid_image,cur_valid_label = sess.run([valid_exam,valid_label])
                    eval_corct = sess.run(eval_step,feed_dict={images:cur_valid_image,labels:cur_valid_label})
                    print('Step %d: Validation accuracy =%.1f%%' % (i,eval_corct*100))
            writer.close()
            coord.request_stop()
            coord.join(threads)



