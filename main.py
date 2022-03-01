import torch
import matplotlib.pyplot as plt
import tempfile
import time
import numpy as np
from torch import optim
from scipy import ndimage
from numpy import random
import myunet3D
import myATunet3D
import myATunet3D_SEAT
import myHighResNet3D
import myVnet
import myResNet3DMedNet
import myDenseVoxelNet
import myDAF3D
import my3d_attention_unet
import mydensesharp
from torchsummary import summary


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MyDiceLoss(torch.nn.Module):
    def __init__(self):
        super(MyDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, sigmoid=True):
        # flatten label and prediction tensors
        if sigmoid:
            inputs = torch.sigmoid(inputs).view(-1)
        else:
            inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1-dice

class MyDice(torch.nn.Module):
    def __init__(self):
        super(MyDice, self).__init__()

    def forward(self, inputs, targets, smooth=1, sigmoid=True):
        # flatten label and prediction tensors
        if sigmoid:
            inputs = torch.sigmoid(inputs).view(-1)
        else:
            inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return dice

class MyComboLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, sigmoid=True):
        ALPHA = 0.8  # < 0.5 penalises FP （false positives) more, > 0.5 penalises FN (false negatives) more
        CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss
        eps = 1e-7

        # # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # flatten label and prediction tensors
        if sigmoid:
            inputs = torch.sigmoid(inputs).view(-1)
        else:
            inputs = inputs.view(-1)

        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        return combo

# PyTorch
class MyDiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyDiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, sigmoid=True):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if sigmoid:
            inputs = torch.sigmoid(inputs).view(-1)
        else:
            inputs = inputs.view(-1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class MyFocalLoss(torch.nn.Module):
    #alpha是控制类别不平衡的.如果 y = 1 类样本个数大于 y = 0，那么 alpha 小于 0.5，保护样本少的类，而多惩罚样本多的类。
    #alpha取比较小的值来降低负样本（多的那类样本）的权重。

    #gamma是控制难易样本的.  目的是通过减少易分类样本的权重，从而使得模型在训练时更专注于难分类的样本。
    #gamma越大，相对容易样本，给予困难样本的权重越大

    def __init__(self, weight=None, size_average=True):
        super(MyFocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.1, gamma=2, smooth=1, sigmoid=True):
        # ALPHA = 0.8
        # GAMMA = 2
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # # flatten label and prediction tensors
        if sigmoid:
            inputs = torch.sigmoid(inputs).view(-1)
        else:
            inputs = inputs.view(-1)

        # inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        # BCE = - ((targets * torch.log(inputs) + ((1.0 - targets) * torch.log(1.0 - inputs))))
        # BCE = BCE.mean(-1)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
        return focal_loss

def rotate(inputs, x):
    return torch.from_numpy(ndimage.rotate(inputs, x, reshape=False))

def Average(lst):
    return sum(lst) / len(lst)

class DataDataset(Dataset):
    def __init__(self, path_data):
        self.img_dir = os.listdir(path_data)
        self.path_data = path_data

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_data, self.img_dir[idx])

        image_mask = np.load(img_path)                #  获取dic类型的image_label，3D image+3D label
        image = image_mask[0].astype(np.float32)
        mask = image_mask[1].astype(np.float32)

        # image = np.clip(image, -300, 300)              #normalization
        # image = (image + 300.0) / 600.
        # mask = np.where(mask > 0, 1, 0)

        # flag = 1
        # for x in range(0, 3):
        #     if flag == 1:
        #         rand = (random.random()*2.-1) * 3.                #角度随机数, rotation
        #         image = ndimage.rotate(image, rand, reshape=False, order=1)
        #         mask = ndimage.rotate(mask, rand, reshape=False, order=1)
        #         flag = flag + 1
        #     elif flag == 2:
        #         noise = np.random.normal(0, 0.01, image.shape)     #添加Gaussian噪声
        #         image = (image + noise).astype(np.float32)
        #         flag = flag + 1
        #     elif flag == 3:                                      #每一列平移
        #         rand = (random.random() * 2. - 1) * 3.
        #         image = ndimage.shift(image, np.array([rand, rand, 0]))
        #         mask = ndimage.shift(mask, np.array([rand, rand, 0]))
        #         flag = flag + 1

        image =np.expand_dims(image, axis=0)           #  扩展为4维
        mask = np.expand_dims(mask, axis=0)          #  扩展为4维

        return {'image': image, 'label': mask}

class Dataset_ct_pet_seg(Dataset):
    def __init__(self, path_data):
        self.img_dir = os.listdir(path_data)
        self.path_data = path_data

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_data, self.img_dir[idx])

        image_mask = np.load(img_path)                #  获取dic类型的image_label，3D image+3D label
        image = image_mask[0:2].astype(np.float32)
        mask = image_mask[2].astype(np.float32)

        # image =np.expand_dims(image, axis=0)           #  扩展为4维
        mask = np.expand_dims(mask, axis=0)          #  扩展为4维

        return {'image': image, 'label': mask}


#---------------------------------------------------------Setup data directory--------------------------------
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
localtime = time.asctime(time.localtime(time.time()))
print(root_dir)
print("start")
print ("current time :", localtime)


#----------------------------------------load liyuan's mat data-------------------------------------------
Train_data_ind = list({2,3,4})
Val_data_ind = list({1})
# [-30, 30], random(x-axis, y-axis, and z-axis)
# Total 2124 (452 nodes+ 1672)  F2: 26p (125nodes, B*3+M*2, 498aug) F3: 26p (157nodes, B*2+M*10, 618aug) F4: 26p (170nodes, B*2+M*10 (delete 8 cases), 556aug)
# 152 nodes  V1: 25 patients
train_path=r'D:/Residual_Disease/Data/ln_seg_data/3D_liyuan_pet_ct/Tr2-4_V1_T5/ct/train'
val_path=r'D:/Residual_Disease/Data/ln_seg_data/3D_liyuan_pet_ct/Tr2-4_V1_T5/ct/val'

train_ds = DataDataset(train_path)    #通过data_mat_load_3Dnumy.py将.mat装成npy
print("Define train_ds")
train_loader = DataLoader(train_ds, batch_size=60, shuffle=True, num_workers=0)        #定义training loader
print("Define train_loader")
val_ds = DataDataset(val_path)
print("Define val_ds")
val_loader = DataLoader(val_ds, batch_size=60, num_workers=0)                       #定义validation loader
print("Define val_loader")

# train_ds = Dataset_ct_pet_seg(train_path)
# print("Define train_ds")
# train_loader = DataLoader(train_ds, batch_size=86, shuffle=True, num_workers=0)        #定义training loader
# print("Define train_loader")
# val_ds = Dataset_ct_pet_seg(val_path)
# print("Define val_ds")
# val_loader = DataLoader(val_ds, batch_size=86, num_workers=0)                       #定义validation loader
# print("Define val_loader")

#--------------------------------------Create Model, Loss, Optimizer-------------------------------------
# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")

# model = myATunet3D.ATUNet3D(in_channels=1, n_classes=1, base_n_filter=10).to(device)              #3d-Unet model;   in_channels=2
model = myATunet3D_SEAT.ATUNet3D(in_channels=1, n_classes=1, base_n_filter=10).to(device)              #3d-Unet model;   in_channels=2
# model = myunet3D.UNet3D(in_channels=1, n_classes=1, base_n_filter=12).to(device)              #3d-Unet model;   in_channels=2
# model = myVnet.VNet(in_channels=1, classes=1).to(device)                                     # Vnet
# model = myHighResNet3D.HighResNet3D(in_channels=1, classes=1).to(device)                     #HighResNet3D
# model = myResNet3DMedNet.ResNetMed3D(in_channels=1, classes=1).to(device)                    #ResNetMed3D
# model = myDenseVoxelNet.DenseVoxelNet(in_channels=1, classes=1).to(device)                   #DenseVoxelNet
# model = myDAF3D.DAF3D().to(device)                                                           #DAF3D--------(A3dunet)
# model = my3d_attention_unet.UNet3D(in_channels=1, out_channels=1).to(device)                 # 3D attention unet------(AE3dunet)
# model = mydensesharp.DenseSharp().to(device)                                                   # 3D dense sharp

pretrain_epoch = 0
# model_load_path=(r'D:/Residual_Disease/3D_Pytorch/MONAI_npy_input/saved_images3_densevoxelnet/500/')
# # model_load = os.path.join(model_load_path, "epoch_{}_metric_model.pth".format(pretrain_epoch+1))
# model_load = os.path.join(model_load_path, "best_metric_model.pth")
# if model_load:
#     model.load_state_dict(torch.load(model_load))
#     print('load pretrain model, and restart to train---------------------------------')
# else:
#     pretrain_epoch = 0               #initialize the pretrain_epoch
#     print('No pretrain model, and restart to train---------------------------------')


loss_function = MyDiceLoss()
# loss_function = MyComboLoss()
# loss_function = MyDiceBCELoss()
# loss_function = MyFocalLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, threshold=0.001)   # Plateaus 的方式，改变learning rate
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 100, 200, 400, 600, 800], gamma=0.5)           # Multi-step 的方式，改变learning rate

my_dice_metric = MyDice()
# dice_metric = DiceMetric(include_background=False, reduction="mean")
# print(model)

print("Create Model, Loss, Optimizer")
#--------------------------Execute a typical PyTorch training process-------------------------------------
max_epochs = 400
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []        #training dice loss
val_dice_loss = []            #记录validation dice loss
metric_values = []
dice_record = []
dice_record_outcome=[]
loss_record = []
loss_record_outcome = []

print("Initialize parameters")

time_stamp = time.time()
save_img_dir = r'D:/Residual_Disease/3D_Pytorch/MONAI_npy_input/saved_images1_3dunet/liyuan_ct/test/'
epoch_dir = f"time_{time_stamp}"
save_model_epoch_dir = os.path.join(save_img_dir, epoch_dir)

for epoch in range(max_epochs):
    print('lr = {}'.format(optimizer.param_groups[0]['lr']))
    if not os.path.exists(save_model_epoch_dir):
        os.makedirs(save_model_epoch_dir)
    else:
        pass
    if not os.path.exists(os.path.join(save_model_epoch_dir, str(epoch+1))):               #每个epoch新建个文件夹
        os.makedirs(os.path.join(save_model_epoch_dir, str(epoch+1)))
    else:
        pass

    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:  # bug，can not enter to next step
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )

        # CUDA_LAUNCH_BLOCKING = 1
        inputs = torch.clip(inputs, -300, 300)
        inputs = (inputs + 300.0) / 600.
        optimizer.zero_grad()                  #zeroes the gradient buffers of all parameters,因为每个训练步骤中我们都想计算新的梯度，而不关心上一批的梯度。不将 grads 归零会导致跨批次的梯度累积。
        outputs = model(inputs)                #calculate output
        # print('outputs', outputs.shape)
        # print('label', labels.shape)
        loss = loss_function(outputs, labels.type(torch.float))   #calculate loss according to comparison with outputs and labels

        loss.backward()                        #calculate the gradient, 计算图grad中所有张量的属性
        optimizer.step()                       #Gardient Descent, optimize model parameter
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")

    epoch_loss /= step                         #epoch_loss = epoch_loss/step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average training loss: {epoch_loss:.4f}")
    np.savetxt(os.path.join(save_model_epoch_dir, 'training_dice_average_loss_epoch.txt'), epoch_loss_values)           #存取不同epoch下的average training loss

    fig = plt.figure(figsize=(20, 20))
    fig_storage_dir = r'D:/Residual_Disease/3D_Pytorch/MONAI_npy_input/saved_images1_3dunet/liyuan_ct/test/'

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            batch_idx = 1
            epoch_loss = 0
            step = 0
            for val_data in val_loader:
                step += 1
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )

                val_labels = torch.where(val_labels > 0, 1, 0)                   #删除空心点
                val_outputs = model(val_inputs)
                loss = loss_function(val_outputs, val_labels.type(torch.float))

                # print('val_labels', val_labels.shape)
                # print('val_outputs', val_outputs.shape)
                # print('lenght_data', len(val_data["image"][0, 0, 0, 0,]))

                epoch_loss += loss.item()
                print(
                    f"{step}/{len(val_ds) // val_loader.batch_size}, "
                    f"val_loss: {loss.item():.4f}")
                loss_record.append(loss.item())

                # compute metric for current iteration
                dice = my_dice_metric(val_outputs, val_labels)           #打印每个val patient的dice
                print("Dice is {}".format(dice))
                dice_record.append(dice.item())

                batch_idx += 1

            # aggregate the final mean dice result
            metric=np.mean(np.array(dice_record))
            loss_value=np.mean(np.array(loss_record))

            # print('val_mean_dice', metric)
            # reset the status for next validation round
            # dice_metric.reset()

            if epoch % 20 == 0:
                torch.save(model.state_dict(), os.path.join(
                    fig_storage_dir, "epoch_{}_metric_model.pth".format(pretrain_epoch+epoch+1)))  #here adds the pretrain_epoch

            # metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(                 #save best model
                    fig_storage_dir, "best_metric_model.pth"))
                print("saved new best metric model")

            scheduler.step((1 - best_metric))                           #此处对应 Plateaus 的方式改变learning rate
            # scheduler.step()                                          #此处对应 multi-step 的方式改变learning rate
            loss_record_outcome.append(loss_value)
            # dice_record_outcome.append(1-metric)
    np.savetxt(os.path.join(save_model_epoch_dir, 'val_dice_average_loss_epoch.txt'), loss_record_outcome)  # 存取不同epoch下的average training loss