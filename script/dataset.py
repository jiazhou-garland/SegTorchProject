import random
import os
import itertools
from PIL import Image as PilImage
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

def sort_images(image_dir, image_type):
    """
    对文件夹内的图像进行按照文件名排序

    image_dir：图像与标签文件夹地址
    image_type：tif or png
    """
    files = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.{}'.format(image_type)) \
                and not image_name.startswith('.'):
            files.append(os.path.join(image_dir, image_name))

    return sorted(files)

def train_val_divide(image_dir, val_ratio=0.2, fold=0):
    """
    训练集与验证集划分，输出txt文件每行格式为元组（图像文件名，标签文件名）

    val_ratio = 0.8 划分训练集与测试集比例8：2
    fold = 0 控制5折交叉验证的训练集与验证集划分，0表示每隔5张的第五张为验证集
    """

    print('开始划分训练集与验证集')
    images = sort_images(image_dir, 'tif')
    labels = sort_images(image_dir, 'png')

    train = []
    val = []
    val_interval = int((1 / val_ratio))

    for i in range(len(images)):
        if ((i+1) == fold or (i+1)%val_interval == fold): #每隔val_interval个划分至val集
            val.append((images[i], labels[i]))
            # print('正划分第{}张图片为验证集'.format(i + 1))
        else:
            train.append((images[i], labels[i]))
            # print('正划分第{}张图片为训练集'.format(i + 1))

    print("训练集与验证集划分结束")

    # with open("./train.txt", "w") as output:
    #     output.write(str(train))
    #     print('保存训练集成功')
    # with open("./val.txt", "w") as output:
    #     output.write(str(val))
    #     print('保存验证集成功')

    return train, val

def get_img_label_paths(image_dir, train_or_val):

    """
    加载训练集或验证集

    if train_or_val == 0: 加载训练集
    if train_or_val == 1 ：加载验证集
    """

    img_seg_pairs = []

    if not os.path.exists('./train.txt') or not os.path.exists('./val.txt'):
        train_val_divide(image_dir, fold=0)

    if train_or_val == 0:
        with open("./train.txt", "r") as f:
            img_seg_pairs = f.read()
            img_seg_pairs = [img_seg_pairs]

    elif train_or_val == 1:
        with open("./val.txt", "r") as f:
            img_seg_pairs = f.read()
            img_seg_pairs = [img_seg_pairs]

    else:
        print('train_or_test should be values of 0 or 1')


    return img_seg_pairs

def transform_label(label, label_form, class_num):
    """
    if label_form = 1:
    [256,256]→[256,256,10] 10分类标签，对于每个channel，值1表示此像素点属于此类别，否则不是

    if label_form = 2:
    [256,256]→[256,256,1] 二分类标签,且标签类别为class_num，对于每个channel，值1表示此像素点属于此类别，否则不是

    if label_form = 3:
    [256,256]→[256,256] 原始数据，类别为1-10
    """

    transformed_label = np.zeros((256, 256, 10))
    for i in range(256):
        for j in range(256):

            index = int(label[i, j])
            transformed_label[i, j, index-1] = 1

    if label_form == 1:
        pass

    elif label_form == 2:
        transformed_label = transformed_label[:,:,class_num-1]

    elif label_form == 3:
        transformed_label = label

    else:
        print("label_form should be values of 1, 2 and 3")

    return transformed_label

def whether_over_sample(label, label_form, class_num, shreshold=10):
    """
    判断标签是否需要过采样
    :param label: 需要判断的标签
    :param shreshold: 判断标准，即除耕地，林地外的其他类，任一标注比例大于此shreshold时，认定为需要过采样
    :return: False 表示不需要过采样， True表示需要
    """

    label = transform_label(label, label_form, class_num)
    label_class_proportion = np.empty(10)
    judgement = False

    for i in range(1,10):  # 检查第2至10类，即耕地，林地除外的其他类

        label_class_proportion[i] = float(label[:, :, i].sum()) / (256 * 256) * 100
        # min_class = np.where(label_class_proportion == np.max(label_class_proportion)) + 2

        if label_class_proportion[i] >= shreshold:
            judgement = True
            break

    return judgement

def transform():
    """
    生成数据增强的list，训练集可能出现的数据增强包括：上下、左右翻转，转置，旋转，加高斯噪声以及随机亮度对比度变化
    :return: 数据增强操作列表
    """
    transform_list = A.Compose([
        A.OneOf([
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.Transpose()
        ]),
        A.RandomRotate90(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            # 高斯噪点
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.RandomBrightnessContrast(p=0.2)])

    return transform_list

def img_label_augmentation(img, label, label_form, class_num, train_or_val):
    """
    数据增强:训练集可能出现的数据增强包括：上下、左右翻转，转置，旋转，加高斯噪声以及随机亮度对比度变化,再做0-1归一化；
            验证集仅做0-1归一化
    :param train_or_val: 0 训练集 ；1 验证集
    :return: 增强图片，增强标签
    """

    label = transform_label(label, label_form, class_num)
    transform_list = transform()

    if train_or_val == 0:
        aug_img = transform_list(image=img, mask=label)['image']
        aug_img = np.divide(aug_img, 255)
        aug_label = transform_list(image=img, mask=label)['mask']
        aug_label = np.divide(aug_label, 255)

    elif train_or_val == 0:
        aug_img = np.divide(img, 255)
        aug_label = np.divide(label, 255)

    else:
        print("train_or_test should be values of 0 or 1")


    return aug_img, aug_label

def image_segmentation_generator(image_dir, batch_size, train_or_val, label_form, class_num=1):
    '''

     图片标签生成器，对判断需要过采样的图片与标签，进行10倍的过采样，且此倍数由概率控制，实际增强次数在9次上下波动；

     判断标准需要过采样的标准：除耕地，林地外的其他类，任一标注比例大于shreshold=10%时，认定为需要过采样

    :param image_dir: 图像与标签文件夹地址
    :param batch_size: batch大小
    :param train_or_val: 训练集 0 测试集 1

    :param label_form:
    if label_form = 1:
    [256,256]→[256,256,10] 10分类标签，对于每个channel，值1表示此像素点属于此类别，否则不是

    if label_form = 2:
    [256,256]→[256,256,1] 二分类标签,且标签类别为class_num，对于每个channel，值1表示此像素点属于此类别，否则不是

    if label_form = 3:
    [256,256]→[256,256] 原始数据，类别为1-10

    :param class_num: 二分类标签的标签类别，1-10 分别表示为"耕地","林地","草地","道路",
                     "城镇建设用地","农村建设用地","工业用地","构筑物","水域","裸地"
                     注意：仅label_form参数为2时，需指定此参数，否则可忽略
    '''

    train, val = train_val_divide(image_dir, val_ratio=0.2, fold=0)

    if train_or_val == 0:
        data = train
    elif train_or_val == 1:
        data = val
    else:
        print('train_or_test should be values of 0 or 1')

    # data = get_img_label_paths(image_dir, train_or_val)
    random.shuffle(data)
    zipped = itertools.cycle(data)


    X = []
    Y = []
    i = 1
    while i <= batch_size:

        img, label = next(zipped)
        img = np.array(PilImage.open(img))
        label = np.array(PilImage.open(label))
        judgement = whether_over_sample(label, label_form, class_num)

        if judgement:
            # print('图片需要增强')
            r = 1 #首次必增强
            while r >= 0.1 and i <= batch_size:
                r = random.random() #是否需要增强下一次由随机数决定
                # print('随机概率为{},图片重复增强'.format(r))
                aug_img, aug_label = img_label_augmentation(img, label, label_form, class_num, train_or_val)
                X.append(aug_img)
                Y.append(aug_label)
                # print(i)
                i = i + 1
        else:
            # print('图片不需要增强')
            X.append(img)
            Y.append(label)
            # print(i)
            i = i + 1

    print(np.array(X).shape, np.array(Y).shape)
    return np.array(X), np.array(Y)

def display(display_list,label_list):
    """
    图片与标签展示
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(3, 4, 1),
    plt.title('原图')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(3, 4, 2),
    plt.title('RGB_image')
    plt.imshow(image)
    plt.axis('off')

    for i in range(10):
        plt.subplot(3, 4, i+3)
        plt.imshow(display_list[i+2], cmap='gray')
        plt.title(label_list[i])
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train_images_path = r"E:\zjz\seg\data\suichang_round1_train_210120"
    train = image_segmentation_generator(train_images_path, batch_size=10,
                                         train_or_val=0, label_form=1)
    img, mask = train
    label_name= ["耕地","林地","草地","道路","城镇建设用地",
                 "农村建设用地","工业用地","构筑物","水域","裸地"]
    for i in range(10):
        label = mask[i,:,:,:]
        # print(label.shape)
        image = img[i,:,:,:]
        # print(image.shape)
        image_RGB = image[:,:,0:3]
        display_list = [image, image_RGB, label[:, :, 0], label[:, :, 1],
                        label[:,:,2], label[:,:,3], label[:,:,4], label[:,:,5],
                        label[:,:,6], label[:,:,7], label[:,:,8], label[:,:,9]]
        display(display_list, label_name)
