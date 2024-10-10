import albumentations as A
from PIL import Image
import numpy as np
import cv2
import os


path = r'C:\Users\49011\Desktop\毕业论文\论文数据\arg_test'
img_ls = os.listdir(path+'/train')
num = len(img_ls)-4800

for i in range(num):
    image = Image.open(path+'/train/{}.png'.format(i))
    train_img = np.array(image) # resize只支持array, 不支持jpg
    label = Image.open(path+'/train_labels/{}.png'.format(i))
    label_img = np.array(label)
    print(train_img.shape, label_img.shape)


    trans1 = A.Compose([

            A.HorizontalFlip(p=1), # 水平翻转
            # A.VerticalFlip(p=1), # 垂直翻转
            # # 平移缩放旋转三个一
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=20,
            #                          interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),

        ])

    # result = trans1(image=train_img, mask=label_img)
    # print(result['image'].shape, result['mask'].shape)
    #
    # t_img = Image.fromarray(result['image'])
    # t_img.save(path+'/train/12.png')
    # l_img = Image.fromarray(result['mask'])
    # l_img.save(path+'/label/12.png')

    trans2 = A.Compose([
            # A.HorizontalFlip(p=1), # 水平翻转
            A.VerticalFlip(p=1), # 垂直翻转
            # 平移缩放旋转三个一
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=20,
            #                             interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),

        ])

    trans3 = A.Compose([
            # A.HorizontalFlip(p=1), # 水平翻转
            # A.VerticalFlip(p=0.1), # 垂直翻转
            # 平移缩放旋转三个一
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=20,
                                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),

        ])

    result = trans3(image=train_img, mask=label_img)
    print(result['image'].shape, result['mask'].shape)

    t_img = Image.fromarray(result['image'])
    t_img.save(path+'/train/{}.png'.format(num*3+i))
    l_img = Image.fromarray(result['mask'])
    l_img.save(path+'/train_labels/{}.png'.format(num*3+i))

    print(i)
