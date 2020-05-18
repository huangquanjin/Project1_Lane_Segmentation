import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

image_list = []
label_list = []

image_dir = 'D:\Study\Pycharm2019\BaiduCV01\data\Image'
label_dir = 'D:\Study\Pycharm2019\BaiduCV01\data\Label'

for s1 in os.listdir(image_dir):
    image_sub_dir1 = os.path.join(image_dir, s1)
    label_sub_dir1 = os.path.join(label_dir, 'label_' + str.lower(s1), 'Label')
    # print(image_sub_dir1, label_sub_dir1)
    for s2 in os.listdir(image_sub_dir1):
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)
        # print(image_sub_dir2)
        # print(label_sub_dir2)
        for s3 in os.listdir(image_sub_dir2):
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)
            for s4 in os.listdir(image_sub_dir3):
                s44 = s4.replace('.jpg', '_bin.png')
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s44)
                # optional
                if not os.path.exists(image_sub_dir4):
                    print(image_sub_dir4)
                    continue
                if not os.path.exists(label_sub_dir4):
                    print(label_sub_dir4)
                    continue
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)
                
assert len(image_list) == len(label_list)

print('The length of image dataset in {}, and label is {}'.format(len(image_list), len(label_list)))

# 存储到csv文件中
total_image = len(image_list)
# 训练集划分：train:validation:test=6:2:2
# 也可以：7:2:1
sixth_part = int(total_image * 0.6)
eight_part = int(total_image * 0.8)

all = pd.DataFrame({'image':image_list, 'label':label_list})
all_shuffle = shuffle(all)

train_dataset = all_shuffle[:sixth_part]
val_dataset = all_shuffle[sixth_part:eight_part]
test_dataset = all_shuffle[eight_part:]

train_dataset.to_csv('D:/Study/Pycharm2019/BaiduCV01/data_list/train.csv', index=False)
val_dataset.to_csv('D:/Study/Pycharm2019/BaiduCV01/data_list/val.csv', index=False)
test_dataset.to_csv('D:/Study/Pycharm2019/BaiduCV01/data_list/test.csv', index=False)

