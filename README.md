# Transfer_learning_Tfrecord
A complete in deep learning project is used for image classification

本项目采用迁移学习（Inception-v3）与tensorflow特有的读写数据格式（Tfrecord）实现对花的识别分类。它是一个完整的项目（约400行代码），其目的是让使用者能够清晰了解tensorflow的使用，我认为它是一个完全学习范式，主要包括写入数据（data_factory.py），读取数据（data_reader.py）,训练迁移学习模型（train.py），测试模型（test.py），运行样例（demo.py）以及包括一些用于图像处理以及数值计算的小组件，它们存储在utilss目录下。

IDE：PYCHRAM
matplotlib==3.3.2
numpy==1.19.0
tensorflow==1.2.0

主程序：main.py

步骤1：数据读写
如果你想重新读取数据，将if_writer参数设置为True，Tfrecord格式数据保存在datasets下，0为训练集，1为验证集，2为测试集/原始数据保存在flower_photos下
如果你不想读取数据则if_writer参数设置为Fals

步骤2：模型训练与测试
模型的权重文件保存在model_factory目录下，你如果想重新训练则需要把if_train设置为True
如果不想训练，想查寻测试的结果则设置为False，模型的效果不太理想，精度只有0.4，主要是作者的电脑读取[299,299,3]的图片时容易导致内存溢出，bacth_size=5且iter=10，epoch=100；如果设备允许可以适当调高上述参数并重新训练。
测试的结果会不稳定，这是因为模型中加入了BN层和dropout层以及训练欠拟合的问题。

步骤3：案例运行
案例运行，作者采用了一个交互式的循环，使用者训练后可以注释掉 step_data(if_write)， step_model()两个函数；输入图片所在的路径名即可得到模型给出的答案。

原始图像数据下载地址：http://download.tensorflow.org/example_image/flower_photos.tgz

Inception_v3模型参数下载地址：http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz 下载后放到model文件夹下

tfrecord格式数据下载地址：链接：https://pan.baidu.com/s/1cSeyS7spOeh2GjU6RxzU8Q 下载后将文件放到目录下
提取码：ty7g

训练模型的权重文件下载地址：链接：https://pan.baidu.com/s/1-uoVDKCgUGoRpYD_Q6RJHA 下载后将文件放到目录下
提取码：74q6
