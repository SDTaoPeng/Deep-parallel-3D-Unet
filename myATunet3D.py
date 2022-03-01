import torch.nn as nn
import torch

class LWBR(nn.Module):
    #---https://github.com/mobarakol/Learning_WhereToLook/blob/master/model.py---------------------------
    def __init__(self, out_c):
        super(LWBR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        x = x + x_res
        return x

class BR(nn.Module):
    #---https://github.com/mobarakol/Learning_WhereToLook/blob/master/model.py---------------------------
    def __init__(self, out_c):
        super(BR, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(out_c))
        self.transposeconv1 = nn.ConvTranspose3d(out_c, out_c, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(out_c))
        self.conv3 =nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x_left = self.conv1(x)
        x_left = self.transposeconv1(x_left)

        x_right = self.conv2(x_left)
        x_right = self.conv3(x_right)
        x = x_left + x_right
        return x

class Dense_residual_block(nn.Module):
    #---https://aapm.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/mp.14617------------
    def __init__(self, out_c):
        super(Dense_residual_block, self).__init__()
        self.conv1 = nn.Conv3d(out_c, out_c, kernel_size=1, padding=0)
        self.conv3 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(out_c, out_c, kernel_size=5, padding=2)

    def forward(self, x):
        x_out1 = self.conv1(x)
        x_out3 = self.conv3(x_out1)
        x_out5 = self.conv5(x_out3 + x)
        x_out = x + x_out1 + x_out3 + x_out5
        return x_out


# class Attention_block(nn.Module):
#     #----https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/AttU-Net.png----------
#     #----https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py----------------
#     def __init__(self, F_g, F_l, F_int):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.InstanceNorm3d(F_int))
#
#         self.W_x = nn.Sequential(
#             nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.InstanceNorm3d(F_int))
#
#         self.psi = nn.Sequential(
#             nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.InstanceNorm3d(1),
#             nn.Sigmoid())
#         self.Sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1+x1)
#         psi = self.psi(psi)
#         psi = self.Sigmoid(psi)
#         out=x*psi
#         return out

class Attention_block(nn.Module):
    #----https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/AttU-Net.png----------
    #----https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py----------------
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int))

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid())
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        print('g1', g1.shape)
        x1 = self.W_x(x)
        print('x1', x1.shape)
        psi = self.relu(g1+x1)
        print('psi', psi.shape)
        psi = self.psi(psi)
        psi = self.Sigmoid(psi)
        out=x*psi
        print('out', out.shape)
        return out

class Skip_scSE1(nn.Module):
    #----https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/AttU-Net.png----------
    def __init__(self, in_c):
        super(Skip_scSE1, self).__init__()
        self.c3 = nn.Conv3d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.c5 = nn.Conv3d(in_c, in_c, kernel_size=5, stride=1, padding=2, bias=True)

        self.pool=nn.MaxPool3d((16, 16, 12))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.c1 = nn.Conv3d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC1 = nn.Conv3d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC2 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, in_c):
        out_c3 = self.c3(in_c)
        out_c5 = self.c5(in_c)
        out_1 = out_c3 + out_c5   #step 1
        out_2 = out_1 + in_c
        out_2 = self.pool(out_2)
        out_2=self.FC1(out_2)
        out_2=self.relu(out_2)
        out_2=self.FC2(out_2)
        out_2=self.sigmoid(out_2)
        out_1_2=out_1*out_2
        out_1_2_3=out_1_2+in_c
        out_4=self.c1(in_c)
        out_4_5=out_4*in_c
        out=out_1_2_3+out_4_5
        return out

class Skip_scSE2(nn.Module):
    #----https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/AttU-Net.png----------
    def __init__(self, in_c):
        super(Skip_scSE2, self).__init__()
        self.c3 = nn.Conv3d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.c5 = nn.Conv3d(in_c, in_c, kernel_size=5, stride=1, padding=2, bias=True)

        self.pool=nn.MaxPool3d((32, 32, 24))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.c1 = nn.Conv3d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC1 = nn.Conv3d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC2 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, in_c):
        out_c3 = self.c3(in_c)
        out_c5 = self.c5(in_c)
        out_1 = out_c3 + out_c5   #step 1
        out_2 = out_1 + in_c
        out_2 = self.pool(out_2)
        out_2=self.FC1(out_2)
        out_2=self.relu(out_2)
        out_2=self.FC2(out_2)
        out_2=self.sigmoid(out_2)
        out_1_2=out_1*out_2
        out_1_2_3=out_1_2+in_c
        out_4=self.c1(in_c)
        out_4_5=out_4*in_c
        out=out_1_2_3+out_4_5
        return out

class Skip_scSE3(nn.Module):
    #----https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/AttU-Net.png----------
    def __init__(self, in_c):
        super(Skip_scSE3, self).__init__()
        self.c3 = nn.Conv3d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.c5 = nn.Conv3d(in_c, in_c, kernel_size=5, stride=1, padding=2, bias=True)

        self.pool=nn.MaxPool3d((64, 64, 48))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.c1 = nn.Conv3d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC1 = nn.Conv3d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC2 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, in_c):
        out_c3 = self.c3(in_c)
        out_c5 = self.c5(in_c)
        out_1 = out_c3 + out_c5   #step 1
        out_2 = out_1 + in_c
        out_2 = self.pool(out_2)
        out_2=self.FC1(out_2)
        out_2=self.relu(out_2)
        out_2=self.FC2(out_2)
        out_2=self.sigmoid(out_2)
        out_1_2=out_1*out_2
        out_1_2_3=out_1_2+in_c
        out_4=self.c1(in_c)
        out_4_5=out_4*in_c
        out=out_1_2_3+out_4_5
        return out

class Skip_scSE4(nn.Module):
    #----https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/AttU-Net.png----------
    def __init__(self, in_c):
        super(Skip_scSE4, self).__init__()
        self.c3 = nn.Conv3d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.c5 = nn.Conv3d(in_c, in_c, kernel_size=5, stride=1, padding=2, bias=True)

        self.pool=nn.MaxPool3d((64, 64, 48))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.c1 = nn.Conv3d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC1 = nn.Conv3d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC2 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, in_c):
        out_c3 = self.c3(in_c)
        out_c5 = self.c5(in_c)
        out_1 = out_c3 + out_c5   #step 1
        out_2 = out_1 + in_c
        out_2 = self.pool(out_2)
        out_2=self.FC1(out_2)
        out_2=self.relu(out_2)
        out_2=self.FC2(out_2)
        out_2=self.sigmoid(out_2)
        out_1_2=out_1*out_2
        out_1_2_3=out_1_2+in_c
        out_4=self.c1(in_c)
        out_4_5=out_4*in_c
        out=out_1_2_3+out_4_5
        return out


class scSE1(nn.Module):
    #----Ref. Squeeze-and-Excitation Networks----------
    def __init__(self, in_c):
        super(scSE1, self).__init__()
        self.pool=nn.MaxPool3d((16, 16, 12))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.FC1 = nn.Conv3d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC2 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, in_c):
        out_1=self.pool(in_c)
        print('out_1', out_1.shape)
        out_1=self.FC1(out_1)
        out_1 = self.relu(out_1)
        print('out_1', out_1.shape)
        out_1=self.FC2(out_1)
        out_1=self.sigmoid(out_1)
        print('out_1', out_1.shape)
        out=in_c+out_1
        print('out_1', out.shape)
        return out

class scSE2(nn.Module):
    #----Ref. Squeeze-and-Excitation Networks----------
    def __init__(self, in_c):
        super(scSE2, self).__init__()
        self.pool=nn.MaxPool3d((32, 32, 24))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.FC1 = nn.Conv3d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC2 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, in_c):
        print('out_2', in_c.shape)
        out_1=self.pool(in_c)
        out_1=self.FC1(out_1)
        print('out_2', out_1.shape)
        out_1 = self.relu(out_1)
        out_1=self.FC2(out_1)
        print('out_2', out_1.shape)
        out_1=self.sigmoid(out_1)
        out=in_c+out_1
        print('out_2', out.shape)
        return out

class scSE3(nn.Module):
    #----Ref. Squeeze-and-Excitation Networks----------
    def __init__(self, in_c):
        super(scSE3, self).__init__()
        self.pool=nn.MaxPool3d((64, 64, 48))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.FC1 = nn.Conv3d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC2 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, in_c):
        out_1=self.pool(in_c)
        out_1=self.FC1(out_1)
        out_1 = self.relu(out_1)
        out_1=self.FC2(out_1)
        out_1=self.sigmoid(out_1)
        out=in_c+out_1
        return out

class scSE4(nn.Module):
    #----Ref. Squeeze-and-Excitation Networks----------
    def __init__(self, in_c):
        super(scSE4, self).__init__()
        self.pool=nn.MaxPool3d((64, 64, 48))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.FC1 = nn.Conv3d(in_c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.FC2 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, in_c):
        out_1=self.pool(in_c)
        out_1=self.FC1(out_1)
        out_1 = self.relu(out_1)
        out_1=self.FC2(out_1)
        out_1=self.sigmoid(out_1)
        out=in_c+out_1
        return out

class ATUNet3D(nn.Module):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    #  https://github.com/black0017/MedicalZooPytorch/tree/master/lib/medzoo
    """
    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(ATUNet3D, self).__init__()
        self.LWBR = LWBR(1)
        self.BR = BR(1)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()                         # revise   LeakyReLU  to ReLU
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)                            # revise


        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)                         # revise

        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)                   # revise

        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)                    # revise

        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)                    # revise

        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.Att1 = Attention_block(self.base_n_filter * 8, self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)
        self.Skip_scSE1 = Skip_scSE1(base_n_filter*4)           # Skip_scSE1
        self.scSE1 = scSE1(base_n_filter * 4)                   # scSE1

        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.Att2 = Attention_block(self.base_n_filter * 4, self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)
        self.Skip_scSE2 = Skip_scSE2(base_n_filter*2)            # Skip_scSE2
        self.scSE2 = scSE2(base_n_filter * 2)                    # scSE2

        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.Att3 = Attention_block(self.base_n_filter* 2, self.base_n_filter* 2, self.base_n_filter* 2)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)
        self.Skip_scSE3 = Skip_scSE3(base_n_filter)              # Skip_scSE3
        self.scSE3 = scSE3(base_n_filter)                        # scSE3

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.Att4 = Attention_block(self.base_n_filter, self.base_n_filter, self.base_n_filter)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.Skip_scSE4 = Skip_scSE4(1)                          # Skip_scSE4
        self.scSE4 = scSE3(1)                                    # scSE4

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.sigmoid = nn.Sigmoid()

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),                     # revise
            nn.LeakyReLU())             # revise             LeakyReLU  to ReLU

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),                        # revise
            nn.LeakyReLU(),            # revise               LeakyReLU  to ReLU
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),            # revise        LeakyReLU  to ReLU
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),                         # revise
            nn.LeakyReLU(),             # revise         LeakyReLU  to ReLU
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),                                           #command out
            nn.LeakyReLU())              # revise          LeakyReLU  to ReLU

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        print('out1', out.shape)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        print('out2', out.shape)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        print('out3', out.shape)
        # Element Wise Summation
        out += residual_1               # Conv3d-6
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)


        # Level 2 context pathway
        out = self.conv3d_c2(out)
        print('out4', out.shape)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        print('out5', out.shape)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        print('out6', out.shape)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        print('out7', out.shape)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        print('out8', out.shape)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        print('out9', out.shape)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        print('out10', out.shape)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        print('out11', out.shape)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        print('out12', out.shape)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out
        print('out12-1', context_4.shape)

        # Level 5
        out = self.conv3d_c5(out)
        print('out13', out.shape)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        print('out14', out.shape)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        print('out15', out.shape)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        print('out16', out.shape)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        print('out17', out.shape)

        # Level 1 localization pathway                    #downsample: out       upsample: context_4
        # out = self.Att1(g=context_4, x=out)               # Attention gate 1   #加在concatenate path那一层，将downsample的结果与upsample的结果进行concatenate
        out = torch.cat([out, context_4], dim=1)
        print('out18', out.shape)
        out = self.conv_norm_lrelu_l1(out)
        print('out19', out.shape)
        out = self.conv3d_l1(out)
        print('out20', out.shape)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        print('out21', out.shape)
        # out = self.Skip_scSE1(out)                              # skip_scSE1
        out = self.scSE1(out)                                     # scSE1

        # Level 2 localization pathway
        # out = self.Att2(g=context_3, x=out)               # Attention gate 2
        out = torch.cat([out, context_3], dim=1)
        print('out22', out.shape)
        out = self.conv_norm_lrelu_l2(out)
        print('out23', out.shape)
        ds2 = out
        out = self.conv3d_l2(out)
        print('out24', out.shape)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        print('out25', out.shape)
        # out = self.Skip_scSE2(out)                              # skip_scSE2
        out = self.scSE2(out)                                     # scSE2

        # Level 3 localization pathway
        # out = self.Att3(g=context_2, x=out)               # Attention gate 3
        out = torch.cat([out, context_2], dim=1)
        print('out26', out.shape)
        out = self.conv_norm_lrelu_l3(out)
        print('out27', out.shape)
        ds3 = out
        out = self.conv3d_l3(out)
        print('out28', out.shape)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        print('out29', out.shape)
        # out = self.Skip_scSE3(out)                               # skip_scSE3
        # out = self.scSE3(out)                                      # scSE3

        # Level 4 localization pathway
        # out = self.Att4(g=context_1, x=out)               # Attention gate 4
        out = torch.cat([out, context_1], dim=1)
        print('out30', out.shape)
        out = self.conv_norm_lrelu_l4(out)
        print('out31', out.shape)
        out_pred = self.conv3d_l4(out)
        print('out32', out_pred.shape)
        # out_pred = self.Skip_scSE4(out_pred)                      # Skip_scSE4
        # out_pred = self.scSE4(out_pred)                             # scSE4

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        print('out33', ds2_1x1_conv.shape)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        print('out34', ds1_ds2_sum_upscale.shape)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        print('out35', ds3_1x1_conv.shape)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        print('out36', ds1_ds2_sum_upscale_ds3_sum.shape)
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)
        print('out37', ds1_ds2_sum_upscale_ds3_sum_upscale.shape)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        print('out38', out.shape)
        seg_layer = out
        print('out39', out.shape)
        # seg_layer = self.LWBR(seg_layer)                        # light-weight BR
        # seg_layer = self.BR(seg_layer)                          # BR
        # seg_layer = self.sigmoid(out)
        return seg_layer


