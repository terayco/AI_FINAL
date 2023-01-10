import time
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from UNET import *
from datasets import CityscapesDataset
from miou import compute_miou,hist_info
from torch.autograd import Variable


#设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default="/root/autodl-tmp/cityscapes", help="directory the Cityscapes dataset is in")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
args = parser.parse_args()


# 加载测试集
test_data = CityscapesDataset(args.datadir, split='test', mode='fine', augment=False)
test_batch = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=6)
print(test_data)



print("构造模型中...")



#加载模型
file_model = './unet.pth'
if os.path.isfile(file_model):
    generator = torch.load(file_model)
else:
    print('请加载正确的模型权重')
    exit()

np.seterr(divide='ignore', invalid='ignore')

print("\n验证模型精度中，请耐心等待...")

per_test_mpa = []
per_test_miou = []
#在测试集上计算mpa和loss
generator.eval()
for idx_batch, (imagergb, label_class, labelrgb) in enumerate(test_batch):
    with torch.no_grad():

        x = Variable(imagergb).cuda()
        y_ = Variable(label_class).cuda()
        y = generator.forward(x)
       
        pred_class = torch.zeros((y.size()[0], y.size()[2], y.size()[3]))
        for idx in range(0, y.size()[0]):
            pred_class[idx] = torch.argmax(y[idx], dim=0).cpu().int()

        pred_class = pred_class.unsqueeze(1).float()
        label_class = label_class.unsqueeze(1).float()
        
        pred_class = pred_class.numpy()
        label_class = label_class.numpy()
        
         #计算miou,mpa
        hist, labeld, correct = hist_info(test_data.num_classes, pred_class, label_class)
        iu, miou, miou_noback, mpa = compute_miou(hist, labeld, correct)

        per_test_mpa.append(mpa)
        per_test_miou.append(miou_noback)

print("测试集:"+" | mPA:" + str(sum(per_test_mpa)/len(per_test_mpa))+" | mIoU:"+str(sum(per_test_miou)/len(per_test_miou)))


