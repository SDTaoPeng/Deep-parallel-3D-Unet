
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import time
import numpy as np
import glob
import myunet3D
import myHighResNet3D
import myVnet
import myResNet3DMedNet
import myDenseVoxelNet
import myDAF3D
import my3d_attention_unet
import mydensesharp

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

def ct_mask_pred_image_review(ct, mask, pred,
                              save_fig_name, view=False,
                              fig_storage_dir=r'D:/Residual_Disease/3D_Pytorch/MONAI/saved_images/'):
    ct_index = [1]
    mask_index = [2]
    pred_index = [3]

    fig = plt.figure(figsize=(20, 20))
    columns = 1
    rows = 3
    fig.subplots_adjust(hspace=0, wspace=0)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i in ct_index:
            img = ct[...]
            img_max = ct.max()
            img_min = ct.min()
            cmap = 'gray'
            if i == 1:
                plt.title(f"ct; min {np.min(ct)}, max {np.max(ct)}")
        elif i in mask_index:
            img = mask[...]
            # assert mask.min() >= 0.0, f"check label min; {mask.min()}"
            # assert mask.max() <= 1.0, f"check label max; {mask.max()}"
            img_max = 1
            img_min = 0
            cmap = 'viridis'
            if i == 2:
                plt.title(f"mask, min {np.min(mask)}, max {np.max(mask)}")
        elif i in pred_index:
            img = pred[...]
            img_max = 1
            img_min = 0
            cmap = 'viridis'
            if i == 3:
                plt.title(f"pred, min {np.min(pred)}, max {np.max(pred)}")

        plt.imshow(img, vmin=img_min, vmax=img_max, cmap=cmap)
        plt.colorbar()
        plt.grid()
    plt.suptitle('CT, Mask, Pred', fontsize=20)
    if view:
        plt.show()
    fig.savefig(os.path.join(fig_storage_dir, save_fig_name))
    plt.close(fig)

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
        return 1 - dice

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

class DataDataset(Dataset):
    def __init__(self, path_data):
        self.img_dir = os.listdir(path_data)
        self.path_data = path_data

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_data, self.img_dir[idx])
        print('img_path',img_path)
        filename = img_path.split('/')[7]  # store file name
        filename = filename.split('_')[0]  # store file name
        print('filename', filename)

        image_mask = np.load(img_path)                #  获取dic类型的image_label，3D image+3D label
        image = image_mask[0].astype(np.float32)
        mask = image_mask[1].astype(np.float32)
        # print(np.any(mask))

        image =np.expand_dims(image, axis=0)           #  扩展为4维
        mask = np.expand_dims(mask, axis=0)          #  扩展为4维
        # mask=np.zeros((256,256,96))

        # image =torch.from_numpy(np.expand_dims(image, axis=0))           #  扩展为4维
        # mask = torch.from_numpy(np.expand_dims(mask, axis=0))          #  扩展为4维

        return {'image': image, 'label': mask, 'filename': filename}

#---------------------------------Setup imports---------------------------------------------------------------
print_config()

#---------------------------------------------------------Setup data directory--------------------------------
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
localtime = time.asctime(time.localtime(time.time()))
print(root_dir)
print("start")
print ("current time :", localtime)

#----------------Define CacheDataset and DataLoader for training and validation-------------------------
val_path=r'D:/Residual_Disease/Data/ln_seg_data/3D_liyuan_pet_ct/Tr2-4_V1_T5/val/'

val_ds = DataDataset(val_path)
print("Define val_ds")
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)                       #定义validation loader
print("Define val_loader")

#--------------------------------------Create Model, Loss, Optimizer-------------------------------------
# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")

model = myunet3D.UNet3D(in_channels=1, n_classes=1, base_n_filter=4).to(device)              #3d-Unet model
# model = myDAF3D.DAF3D().to(device)                                                           #DAF3D--------(A3dunet)
# model = my3d_attention_unet.UNet3D(in_channels=1, out_channels=1).to(device)                 # 3D attention unet------(AE3dunet)

model.load_state_dict(torch.load(r'D:/Residual_Disease/3D_Pytorch/MONAI_npy_input/saved_images1_3dunet/liyuan_ct_pet/best_metric_model.pth'))

# loss_function = DiceLoss(to_onehot_y=True, softmax=True)
# dice_metric = DiceMetric(include_background=False, reduction="mean")
my_dice_metric = MyDice()

print("model.load_state_dict")
model.eval()
print("model.eval")

time_stamp = time.time()

save_img_dir = r'D:/Residual_Disease/3D_Pytorch/MONAI_npy_input/saved_images1_3dunet/liyuan_ct_pet'
epoch_dir = f"current_time_{time_stamp}"
save_model_epoch_dir = os.path.join(save_img_dir, epoch_dir)

dice_record = []
dice_voxel_record = []
filename_dice_voxel_record = []
LN_size=[]

for epoch in range(1):
    if not os.path.exists(save_model_epoch_dir):
        os.makedirs(save_model_epoch_dir)
    else:
        pass
    if not os.path.exists(os.path.join(save_model_epoch_dir, str(epoch+1))):
        os.makedirs(os.path.join(save_model_epoch_dir, str(epoch+1)))
    else:
        pass

    val_dice_loss=[]
    if (epoch + 1) % 1 == 0:
        model.eval()
        with torch.no_grad():
            batch_idx = 1
            epoch_loss = 0
            step = 0
            for val_data in val_loader:
                scan_number=0
                step += 1
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device)
                )

                filename_1 = val_data["filename"][0]
                filename_1=int(filename_1)
                print('filename', filename_1)

                filename_temp=[filename_1]
                filename_temp = np.array(filename_temp)
                filename_dice_voxel_record = np.concatenate((filename_dice_voxel_record, filename_temp), axis=0)

                val_inputs = torch.clip(val_inputs, -300, 300)
                val_inputs = (val_inputs + 300.0) / 600.
                val_labels = torch.where(val_labels > 0, 1, 0)
                val_outputs = model(val_inputs)

                # val_labels = val_labels[:, :, :, :, 2:14]
                # val_outputs = val_outputs[:, :, :, :, 2:14]
                # # print('val_labels.shape', val_labels.shape)
                # p1d = (2, 2)
                # val_labels = torch.nn.functional.pad(val_labels, p1d, "constant", 0)
                # val_outputs = torch.nn.functional.pad(val_outputs, p1d, "constant", 0)
                # print('val_labels.shape', val_labels.shape)

                dice = my_dice_metric(val_outputs, val_labels)
                # dice_numpy = my_dice_metric(val_outputs, val_labels)
                print("Dice is {}".format(dice))
                dice_record.append(dice)

                Each_LN_size=[]
                # dice_voxel_number=[]           #记录每个病人的dice和voxel
                # dice_voxel_record.append(dice_numpy.detach().cpu().numpy())   #记录每个病人的dice和voxel

                sum_number = np.sum(val_labels.detach().cpu().numpy())                                   # 每个voxel的体积，单位是1mm*1mm*1mm
                LN_size_1=[sum_number]
                LN_size_1 = np.array(LN_size_1)
                LN_size = np.concatenate((LN_size, LN_size_1), axis=0)

                if np.random.randint(1000)>0:
                    print("we are going to save some valiadation image")

                    dice_voxel_patient_record = []
                    dice_voxel_patient_record.append(dice.detach().cpu().numpy())

                    print('sum_number', sum_number)
                    dice_voxel_patient_record.append(sum_number)
                    print('dice_voxel_patient_record', dice_voxel_patient_record)
                    dice_voxel_record=np.concatenate((dice_voxel_record, dice_voxel_patient_record), axis=0)
                    print('dice_voxel_number', dice_voxel_record)

                    filename_dice_voxel_record=np.concatenate((filename_dice_voxel_record, dice_voxel_patient_record), axis=0)

                    LN_number = 0                                        #不同LN的个数
                    for slice_number in range(len(val_data["image"][0, 0, 0, 0,])):
                        image_ct = val_inputs[0, 0, :, :, slice_number].detach().cpu().numpy()
                        image_mask = val_labels[0, 0, :, :, slice_number].detach().cpu().numpy()
                        val_outputs_pre=torch.round(torch.sigmoid(val_outputs))[0, 0,..., slice_number].detach().cpu().numpy()

                        # ct_mask_pred_image_review(image_ct, image_mask, val_outputs_pre,
                        #                           save_fig_name=f"val_pred_{filename_1}_{slice_number}.png",
                        #                           view=False,
                        #                           fig_storage_dir=os.path.join(save_model_epoch_dir,str(epoch+1)))
                batch_idx=batch_idx+1

                dice_record_out = [x.detach().cpu().numpy() for x in dice_record]
                np.savetxt(os.path.join(save_model_epoch_dir, 'dice_record.txt'),
                           dice_record_out, fmt='%.5f')  # 存取不同epoch下的average training loss

                dice_voxel_record_out = [x for x in dice_voxel_record]
                np.savetxt(os.path.join(save_model_epoch_dir, 'dice_voxel_record.txt'),
                           dice_voxel_record_out, fmt='%.5f', newline='\n')  # 存取不同epoch下的average training loss

                np.savetxt(os.path.join(save_model_epoch_dir, 'dice_voxel_size.txt'),
                        LN_size, newline='\n',fmt='%.1f', delimiter='\t')  # 存取不同epoch下的average training loss

                filename_dice_voxel_record_out = [x for x in filename_dice_voxel_record]
                np.savetxt(os.path.join(save_model_epoch_dir, 'filename_dice_voxel_record_out.txt'),
                           filename_dice_voxel_record_out, fmt='%.5f', newline='\n')  # 存取不同epoch下的average training loss


