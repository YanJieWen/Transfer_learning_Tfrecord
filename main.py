# @Time    : 2022/3/1 10:00
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : main
# @Project Name :迁移学习+tfrecord案例
from datas.data_factory import *
from model.train import *
from datas.data_reader import *
from model.test import *
from model.demo import *
#_______________________________________________________________________________________________________________________
#指定参数
input_data = './flower_photos'
train_pet = 0.8
valid_pet = 0.1
num_shard = 3
if_write = False#当为真的时候会存储数据为Tfrecord,它会替换原来的初始数据
w = 299
h = 299
all_label = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
ckpt_file = './model/inception_v3.ckpt'
lr = 0.001
epoch = 100
iter = 10
n_class = len(all_label)
log_dir ='./log'
batch_size = 5
train_file = './model_factory/model.ckpt'
if_train = False#是否需要训练
meta_file ='./model_factory/model.ckpt-99.meta'
model_fac = './model_factory'
#_______________________________________________________________________________________________________________________
#判断是写入数据还是加载数据
def step_data(if_write):
    while True:
        if if_write==True:
            Write_data = To_TfRecord(input_data, train_pet, valid_pet, num_shard,all_label)
            Write_data.get_tfrecord()
            if_write = False
            print('数据已经保存为Tfrecord格式！')
        else:
            print('请读取数据！')
        break
#_______________________________________________________________________________________________________________________
#训练阶段&测试阶段
def step_model():
    while True:
        if if_train:
            trainner = Train_model(train_file,ckpt_file,lr,epoch,iter,n_class,log_dir,w,h,batch_size)
            trainner.train_setting()
        else:
            tester = Test_model(meta_file,model_fac,h,w,batch_size+50)
            pred,cur_test_label = tester.test_setting()
            eval = test_eval(pred,cur_test_label)
            print('预测精度（准确率）为：',eval)
        break
#_______________________________________________________________________________________________________________________
#运行demo
def step_demo():#每次测试结果不一样是因为BN层和DROPOUT层的存在，且训练没有充分！
    transfer_label = ['雏菊','蒲公英','玫瑰花','向日葵','郁金香']
    while True:
        pc_file = input('请输入您的样例图片的路径:')
        label_,image_value = demo(pc_file,h,w,meta_file,model_fac,transfer_label)
        print('这是一朵鲜艳的{}'.format(label_))
        visial_label(image_value)



def main():
    # step_data(if_write)#数据处理的工作
    # step_model()#数据训练和测试的工作
    step_demo()#运行demo的工作


if __name__ == '__main__':
    main()