import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as v_utils
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from UNET import *
from miou import hist_info, compute_miou
from datasets import CityscapesDataset
from tqdm import tqdm
from time import time


# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default="/root/autodl-tmp/cityscapes", help="directory the Cityscapes dataset is in")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
args = parser.parse_args()


# 设置超参数

lr = 0.002

epochs = 100


# 加载训练集集
train_data = CityscapesDataset(args.datadir, split='train', mode='fine', augment=True)
train_batch = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=6)
print(train_data)
#加载验证集
val_data = CityscapesDataset(args.datadir, split='val', mode='fine', augment=False)
val_batch = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=6)

# 定义损失函数
loss_func = nn.CrossEntropyLoss().cuda()
num_classes = train_data.num_classes #训练类别数

#记录训练损失
writer = SummaryWriter('./logs_train')

# initiate generator and optimizer
print("-------------正在构造网络-------------")

generator = U_Net(3,train_data.num_classes)
generator = generator.cuda()
optimizer = torch.optim.AdamW(generator.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 加载之前训练的权重继续训练
# file_model = './unet.pth'
# if os.path.isfile(file_model):
#     generator = torch.load(file_model)
#     print("---从上一次训练的模型加载继续训练---")



# 保存结果
if not os.path.exists('./result/'):
    os.makedirs('./result/')
    
#保存模型参数    
if not os.path.exists('./models/'):
    os.makedirs('./models/bestmodels/')
    
#保存模型参数    
if not os.path.exists('./logs_train/'):
    os.makedirs('./logs_train/')   
    
#保存训练曲线   
if not os.path.exists('./images/'):
    os.makedirs('./images/')  

# 开始训练
train_losses = []
train_time = []
train_mpa = []
train_miou = []
val_losses = []
val_miou = []
val_mpa = []
min_loss = 100
max_mpa = 0
min_loss_epoch = 0
max_mpa_epoch = 0
iu_total = np.zeros((20,))
print("-------------开始训练-----------------")
for epoch in range(epochs):
    generator.train()
    start_time = time()
    epoch_loss = []
    for idx_batch, (imagergb, labelmask, labelrgb) in tqdm(enumerate(train_batch), total=round(train_data.__len__()/args.batch_size),
                                                         colour='green',unit='batch',mininterval=1):
        #前向传播前将梯度置零
        optimizer.zero_grad()
        # 使用GPU进行训练
        x = Variable(imagergb).cuda()
        y_ = Variable(labelmask).cuda()
        y = generator.forward(x)
        #因为使用交叉熵损失，所以要squeeze label
        #这是因为它要求具有形状为[N，W，H]的图像，在2D图像中，对应于该像素应为0<pix[u，v]<classes的类
        y_ = torch.squeeze(y_)

        # 计算loss并反向传播
        loss = loss_func(y, y_)
        epoch_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        

        # 每训练一个epoch保存一次图片，并存储新的模型参数
        # 打印本轮训练信息
        if idx_batch == round(train_data.__len__()/args.batch_size)-1:

            end_time = time()
            runtime = end_time - start_time
            train_time.append(runtime)

            loss = sum(epoch_loss)/len(epoch_loss)

            # 保存相应图片
            v_utils.save_image(x.cpu().data, "./result/original_image_{}_{}.png".format(epoch+1, idx_batch))
            v_utils.save_image(labelrgb.float()/255, "./result/label_image_{}_{}.png".format(epoch+1, idx_batch))

            y_threshed = torch.zeros((y.size()[0], 3, y.size()[2], y.size()[3]))
            
            #将模型预测的最高概率的结果可视化为rgb图片
            for idx in range(0, y.size()[0]):
                maxindex = torch.argmax(y[idx], dim=0).cpu().int()
                y_threshed[idx] = train_data.class_to_rgb(maxindex)

            v_utils.save_image(y_threshed/255, "./result/predict_image_{}_{}.png".format(epoch+1, idx_batch))

            #保存loss最小的模型,并记录最小损失
            if loss <= min_loss:
                torch.save(generator, './models/bestmodels/minLoss_model.pth')
                min_loss = loss
                print("\nepoch = " + str(epoch + 1) + " | loss = " + str(loss) + " | minloss = " + str(min_loss) +
                  " | lr = " + str(scheduler.get_last_lr()[0]) + " | runtime = "+str(end_time-start_time)+"(秒)")
                print('已保存loss最小的模型:epoch = {},loss = {}'.format(epoch + 1, loss))
                min_loss_epoch = epoch + 1
            else:
                print("\nepoch = " + str(epoch + 1) + " | loss = " + str(loss) + " | minloss = " + str(min_loss) +
                  " | lr = " + str(scheduler.get_last_lr()[0]) + " | runtime = "+str(end_time-start_time)+"(秒)")
                
            #记录学习率衰减情况
            writer.add_scalar('train_loss', loss, epoch+1)
            writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch+1)
    # 每隔10个epoch在验证集上验证模型精度和loss，并保存模型,同时在训练集上计算mpa
    if (epoch+1) % 10 == 0 :
        per_train_mpa = []
        per_train_miou = []
        generator.eval()
        print("----------正在计算训练集上的mPA，验证集上的mPA和loss------------")
        for idx_batch, (imagergb, label_class, labelrgb) in enumerate(train_batch):
            with torch.no_grad():
                x = Variable(imagergb).cuda()
                y_ = Variable(label_class).cuda()
                y = generator.forward(x)
                y_ = torch.squeeze(y_)
                
                pred_class = torch.zeros((y.size()[0], y.size()[2], y.size()[3]))
                for idx in range(0, y.size()[0]):
                    pred_class[idx] = torch.argmax(y[idx], dim=0).cpu().int()

                # unsqueese 之后获得形状 [N, 1, W, H] 的图片
                pred_class = pred_class.unsqueeze(1).float()
                label_class = label_class.unsqueeze(1).float()

                pred_class = pred_class.numpy()
                label_class = label_class.numpy()
                #计算累积的总混淆矩阵
                hist, labeld, correct = hist_info(train_data.num_classes, pred_class, label_class)

                iu, miou, miou_noback, mpa = compute_miou(hist, labeld, correct)
                
                per_train_mpa.append(mpa)
                per_train_miou.append(miou_noback)

        train_losses.append(loss)
        train_miou.append(sum(per_train_miou)/len(per_train_miou))
        train_mpa.append(sum(per_train_mpa)/len(per_train_mpa))
        writer.add_scalar('train_mpa', sum(per_train_mpa)/len(per_train_mpa), epoch+1)
        writer.add_scalar('train_mIoU', sum(per_train_miou)/len(per_train_miou), epoch + 1)
        print("epoch = %d" % (epoch + 1))
        print("训练集: | mPA = %.4f" % (sum(per_train_mpa)/len(per_train_mpa)) +"| mIoU = %.4f" % (sum(per_train_miou)/len(per_train_miou))
             +"| loss = %.4f" %(loss))

        val_loss = []
        per_val_mpa = []
        per_val_miou = []
        #在验证集上计算mpa和loss
        for idx_batch, (imagergb, label_class, labelrgb) in enumerate(val_batch):
            with torch.no_grad():
    
                x = Variable(imagergb).cuda()
                y_ = Variable(label_class).cuda()
                y = generator.forward(x)
                y_ = torch.squeeze(y_)

                per_val_loss = loss_func(y, y_).item()
                

                pred_class = torch.zeros((y.size()[0], y.size()[2], y.size()[3]))
                for idx in range(0, y.size()[0]):
                    pred_class[idx] = torch.argmax(y[idx], dim=0).cpu().int()

                pred_class = pred_class.unsqueeze(1).float()
                label_class = label_class.unsqueeze(1).float()

                pred_class = pred_class.numpy()
                label_class = label_class.numpy()
                #计算累积的总混淆矩阵
                hist, labeld, correct = hist_info(train_data.num_classes, pred_class, label_class)

                iu, miou, miou_noback, mpa = compute_miou(hist, labeld, correct)
                iu_total += iu
                
                val_loss.append(per_val_loss)
                per_val_mpa.append(mpa)
                per_val_miou.append(miou_noback)
                
        val_miou.append(sum(per_val_miou)/len(per_val_miou))
        val_mpa.append(sum(per_val_mpa)/len(per_val_mpa))
        val_losses.append(sum(val_loss)/len(val_loss))
        #记录验证集上的loss
        writer.add_scalar('val_loss', sum(val_loss)/len(val_loss), epoch+1)
        writer.add_scalar('val_mpa', sum(per_val_mpa)/len(per_val_mpa), epoch+1)
        writer.add_scalar('val_miou', sum(per_val_miou)/len(per_val_miou), epoch+1)
        torch.save(generator, './models/model_epoch_{}.pth'.format(epoch+1))

        print("验证集: | mPA = %.4f" % (sum(per_val_mpa)/len(per_val_mpa)) + "| mIoU = %.4f" % (sum(per_val_miou)/len(per_val_miou))
              +"| loss = %.4f" %(sum(val_loss)/len(val_loss)))
        
        if sum(per_val_mpa)/len(per_val_mpa) > max_mpa:
            torch.save(generator, './models/bestmodels/maxmPA_model.pth')
            print('已保存mPA最高的模型:epoch = {},mPA = {}'.format(epoch + 1, sum(val_mpa)/len(val_mpa)))
            max_mpa = sum(per_val_mpa)/len(per_val_mpa)
            max_mpa_epoch = epoch + 1

   #每隔一个epoch更新一下scheduler的step
    scheduler.step()

avg_iu = iu_total / (500 * 10)
print("各类别的iu指数为:",avg_iu)
print("训练中最小损失为:{}".format(min_loss))
print("每个epoch平均训练时间 = %.2f(秒)" %(sum(train_time)/len(train_time)))
print("loss最小的epoch:{}".format(min_loss_epoch))
print("mPA最高的epoch:{}".format(max_mpa_epoch)) 

#用matplotlib绘制训练loss曲线
plt.figure()             # 设置图片信息 
plt.plot(train_losses,'dodgerblue',label = 'train')
plt.plot(val_losses,'cyan',label = 'val')
plt.ylabel('loss')
plt.xlabel('epoch')
label=['10','20','30','40','50','60','70','80','90','100']
tick = range(0,10)
plt.xticks(ticks=tick,labels=label)
plt.legend()        #个性化图例（颜色、形状等）
plt.savefig('./images/loss.png') #保存图片
#绘制mpa曲线
plt.figure()             # 设置图片信息 
plt.plot(train_mpa,'dodgerblue',label = 'train')
plt.plot(val_mpa,'orange',label = 'val')
plt.ylabel('mPA')
plt.xlabel('epoch')
label=['10','20','30','40','50','60','70','80','90','100']
tick = range(0,10)
plt.xticks(ticks=tick,labels=label)
plt.legend()        #个性化图例（颜色、形状等）
plt.savefig('./images/mpa.png') #保存图片 
#绘制miou曲线
plt.figure()             # 设置图片信息 
plt.plot(train_miou,'dodgerblue',label = 'train')
plt.plot(val_miou,'orange',label = 'val')
plt.ylabel('mIoU')
plt.xlabel('epoch')
label=['10','20','30','40','50','60','70','80','90','100']
tick = range(0,10)
plt.xticks(ticks=tick,labels=label)
plt.legend()        #个性化图例（颜色、形状等）
plt.savefig('./images/miou.png') #保存图片 



