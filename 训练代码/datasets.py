import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class CityscapesDataset(Dataset):

    def __init__(self, root, split='train', mode='fine', augment=False):

        self.root = os.path.expanduser(root)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.split = split
        self.augment = augment
        self.images = []
        self.targets = []
        self.mapping = {
            0: 0,  # unlabeled
            1: 1,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 2,  # road
            8: 3,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 4,  # building
            12: 5,  # wall
            13: 6,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 7,  # pole
            18: 0,  # polegroup
            19: 8,  # traffic light
            20: 9,  # traffic sign
            21: 10,  # vegetation
            22: 11,  # terrain
            23: 12,  # sky
            24: 13,  # person
            25: 14,  # rider
            26: 15,  # car
            27: 16,  # truck
            28: 17,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 0,  # train
            32: 18,  # motorcycle
            33: 19,  # bicycle
            -1: 0  # license plate
        }
        self.mappingrgb = {
            0: (0, 0, 0),  # unlabeled
            1: (0, 0, 0),  # ego vehicle
            2: (0, 0, 0),  # rect border
            3: (0, 0, 0),  # out of roi
            4: (0, 0, 0),  # static
            5: (111, 74, 0),  # dynamic
            6: (81, 0, 81),  # ground
            7: (128, 64, 128),  # road
            8: (244, 35, 232),  # sidewalk
            9: (250, 170, 160),  # parking
            10: (230, 150, 140),  # rail track
            11: (70, 70, 70),  # building
            12: (102, 102, 156),  # wall
            13: (190, 153, 153),  # fence
            14: (180, 165, 180),  # guard rail
            15: (150, 100, 100),  # bridge
            16: (150, 120, 90),  # tunnel
            17: (153, 153, 153),  # pole
            18: (153, 153, 153),  # polegroup
            19: (250, 170, 30),  # traffic light
            20: (220, 220, 0),  # traffic sign
            21: (107, 142, 35),  # vegetation
            22: (152, 251, 152),  # terrain
            23: (70, 130, 180),  # sky
            24: (220, 20, 60),  # person
            25: (255, 0, 0),  # rider
            26: (0, 0, 255),  # car
            27: (0, 0, 70),  # truck
            28: (0, 60, 100),  # bus
            29: (0, 0, 90),  # caravan
            30: (0, 0, 110),  # trailer
            31: (0, 80, 100),  # train
            32: (0, 0, 230),  # motorcycle
            33: (119, 11, 32),  # bicycle
            -1: (0, 0, 142)  # license plate
        }

        #训练类别数
        self.num_classes = 20

        
        if mode not in ['fine', 'coarse']:
            raise ValueError('无效mode!')
        if mode == 'fine' and split not in ['train', 'test', 'val','predict']:
            raise ValueError('无效mode! 请在"train", "test" "val"中选择')
        if self.split != 'predict':
            if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
                raise RuntimeError('数据集文件目录错误，请检查!')

       #读取训练图片路径
        if self.split != 'predict':
            for city in os.listdir(self.images_dir):
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                for file_name in os.listdir(img_dir):
                    self.images.append(os.path.join(img_dir, file_name))
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], '{}_labelIds.png'.format(self.mode))
                    self.targets.append(os.path.join(target_dir, target_name))
    #打印dataset信息
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '      Number of images: {}\n'.format(self.__len__())
        fmt_str += '      Split: {}\n'.format(self.split)
        fmt_str += '      Mode: {}\n'.format(self.mode)
        fmt_str += '      Augment: {}\n'.format(self.augment)
        fmt_str += '      Root Location: {}\n'.format(self.root)
        return fmt_str


    def __len__(self):
        return len(self.images)

    def mask_to_class(self, mask):
        '''
        将mask通过mapping转为class
        '''
        maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            maskimg[mask == k] = self.mapping[k]
        return maskimg

    def mask_to_rgb(self, mask):
        '''
       将mask转为彩色图像
        '''
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mappingrgb:
            rgbimg[0][mask == k] = self.mappingrgb[k][0]
            rgbimg[1][mask == k] = self.mappingrgb[k][1]
            rgbimg[2][mask == k] = self.mappingrgb[k][2]
        return rgbimg

    def class_to_rgb(self, mask):
        '''
        将类别转为对应彩色图
        '''
        mask2class = dict((v, k) for k, v in self.mapping.items())
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in mask2class:
            rgbimg[0][mask == k] = self.mappingrgb[mask2class[k]][0]
            rgbimg[1][mask == k] = self.mappingrgb[mask2class[k]][1]
            rgbimg[2][mask == k] = self.mappingrgb[mask2class[k]][2]
        return rgbimg

    def __getitem__(self, index):

        #加载图片
        image = Image.open(self.images[index]).convert('RGB')

        #加载标签
        target = Image.open(self.targets[index]).convert('L')
        #自定义transform，将图片转为tensor并归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        #自定义数据增强
        if self.augment:
            # Resize
            image = TF.resize(image, size=(256, 512), interpolation=InterpolationMode.BILINEAR)
            target = TF.resize(target, size=(256, 512), interpolation=InterpolationMode.NEAREST)
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 512))
            image = TF.crop(image, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)
        else:
            # Resize
            image = TF.resize(image, size=(256, 512), interpolation=InterpolationMode.BILINEAR)
            target = TF.resize(target, size=(256, 512), interpolation=InterpolationMode.NEAREST)

        # 转为tensor
        target = torch.from_numpy(np.array(target, dtype=np.uint8))
        image = transform(image)

        # 将label转为对应class
        targetrgb = self.mask_to_rgb(target)
        targetmask = self.mask_to_class(target)
        targetmask = targetmask.long()
        targetrgb = targetrgb.long()


        # 返回路径
        if self.split != 'predict':
            return image, targetmask, targetrgb
        else:
            return None
