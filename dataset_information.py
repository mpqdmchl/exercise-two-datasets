"""
this file's location is : utils/dataset_information.py
dataset :'Thermal Dog and People' and 'MaskWearing'
dataset location: 'datasets/Thermal_DP/train' and 'datasets/MaskWearing/train'

"""

import glob
import os
import numpy
import cv2


def dataset_info():
    current_path = os.getcwd()
    # dataset_path = str(current_path).replace('utils','datasets/Thermal_DP/train')
    dataset_path = str(current_path).replace('utils', 'datasets/MaskWearing/train')

    img_path = os.path.join(dataset_path,'images')
    label_path = os.path.join(dataset_path,'labels')

    # 统计图片信息
    img_files = glob.glob(img_path + r"/*.jpg")
    img_num = len(img_files)
    print('num of images:{}'.format(img_num))
    # img_files = glob.iglob(img_path + r"/*.jpg")
    # for jpg in img_files:
    #     img = cv2.imread(jpg)
    #     w,h,c = img.shape

    # 统计标签信息
    label_class = [0,0,0]
    label_map = []
    label_files=glob.iglob(label_path + r"/*.txt")
    for txt in label_files:
        # print(txt)
        with open(txt,'r') as f:
           # line = f.readlines()
           # print(line)
           for per_line in f.readlines():
                per_label = per_line.strip('\n')
                per_class = int(str(per_label).split(' ')[0])
                per_coordinate= (str(per_label).split(' ')[1:5])
                [normal_w,normal_h] = per_coordinate[-2:]

                normal_area = float(normal_w) * float(normal_h)
                item = (txt,per_class,416,tuple(per_coordinate),normal_area)
                label_map.append(item)
                # print(item)
                # print(per_coordinate)
                if per_class == 0:
                    label_class[0] += 1
                elif per_class == 1:
                    label_class[1] += 1
                else:
                    label_class[2] += 1

    print("[class_0,class_1,else]:",label_class)
    print('num of labels:{}'.format(label_class[0]+label_class[1]+label_class[2]))
    label_array = numpy.array(label_map,dtype=object)

    # 将不同类别的ndarray分开
    array_class0 = label_array[label_array[:, 1] == 0]
    array_class1 = label_array[label_array[:, 1] == 1]

    # 0类的标签面积最值的索引
    erea_min_idx_0 = array_class0[:, 4].argmin()
    erea_max_idx_0 = array_class0[:, 4].argmax()

    # 1类的标签面积最值的索引
    erea_min_idx_1 = array_class1[:, 4].argmin()
    erea_max_idx_1 = array_class1[:, 4].argmax()

    # 展示4个标签
    show_class = [array_class0,array_class0,array_class1,array_class1]
    show_labels = [erea_min_idx_0,erea_max_idx_0,erea_min_idx_1,erea_max_idx_1]

    for array_class,erea_idx in zip(show_class,show_labels):
        show_4_labels(array_class,erea_idx)


def nxywh2xyxy(x_center,y_center,w,h,size):
    x_min = int(x_center * size - (w * size) / 2)
    y_min = int(y_center * size - (h * size) / 2)
    x_max = int(x_center * size + (w * size) / 2)
    y_max = int(y_center * size + (h * size) / 2)

    return x_min,y_min,x_max,y_max


def show_4_labels(array_class,erea_idx):
    # 还原坐标
    x = float(array_class[:, 3][erea_idx][0])
    y = float(array_class[:, 3][erea_idx][1])
    w = float(array_class[:, 3][erea_idx][2])
    h = float(array_class[:, 3][erea_idx][3])
    # img_min_path = str(array_class0[:,0][erea_min_idx_0]).replace('.txt','.jpg')
    _path = str(array_class[:, 0][erea_idx]).replace('/labels/', '/images/')
    img_show_path = _path.replace('.txt','.jpg')

    x_left, y_left, x_right, y_right = nxywh2xyxy(x, y, w, h, 416)
    print(x_left, y_left, x_right, y_right)
    print(img_show_path)

    picture = cv2.imread(img_show_path)
    cv2.rectangle(picture, (x_left, y_left), (x_right, y_right), (0, 0, 255), 3)
    cv2.imshow('img', picture)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dataset_info()
