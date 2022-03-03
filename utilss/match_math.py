# @Time    : 2022/3/1 13:58
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : match_math
# @Project Name :迁移学习+tfrecord案例
#_______________________________________________________________________________________________________________________
#该文件存储一些正则化匹配和数值计算的组件
import re
import numpy as np
def re_match(name):
    return re.findall('\w/(.*?)/\d', name)[0]

def transfer_label(label,all_label):
    for index,value in enumerate(all_label):
        if value==label:
            return int(index)

def test_eval(pred,test_label):
    index_label = np.argmax(pred,axis=1)
    corrct_count = 0
    for index,value in enumerate(test_label):
        if index_label[index]==value:
            corrct_count+=1
    return corrct_count/len(test_label)
