import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

# block operations
class BasicBlock(LightningModule):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,padding=0, act=nn.ReLU):
        super().__init__()
        conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        activation = act()
        bn = nn.BatchNorm2d(out_channels)
        self.block = nn.Sequential(conv, activation, bn)
        
    def forward(self, x):
        return self.block(x)

# https://peiyihung.github.io/mywebsite/category/learning/2021/08/22/build-resnet-from-scratch-with-pytorch.html
class ResidualBlock(nn.Module):
    def __init__(self, ni: int, nf: int):
        
        super().__init__()
        
        # shorcut: If True, assume downsampling is needed
        if ni < nf: 
            
            # change channel size
            self.shortcut = nn.Sequential(
                nn.Conv2d(ni, nf, kernel_size=1, stride=2),
                nn.BatchNorm2d(nf))
            
            # downsize the feature map
            first_stride = 2
        else:
            self.shortcut = lambda x: x
            first_stride = 1
        
        # convnet
        self.conv = nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=3, stride=first_stride, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf)
        )
    
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

# https://github.com/ZQPei/deep_sort_pytorch/blob/master/deep_sort/deep/original_model.py
#  repeat residual layers
def rep_res_layers(c_in,c_out,n_reps):
    """
    :obj: repeat residual block layers
    :param c_in: input channel for first residual block
    :param c_out: output channels for all residual blocks
    :param n_reps: number of times to repeat residual block
    """
    blocks = []
    for i in range(n_reps):
        if i ==0:
            blocks += [ResidualBlock(c_in,c_out),]
        else:
            blocks += [ResidualBlock(c_out,c_out),]
    return nn.Sequential(*blocks)


# norm block
class Norm(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
    def forward(self, x):
        return x.div(x.norm(p=self.p,dim=1,keepdim=True))


class ReID_Architectures(LightningModule):
    
    
    def init_archv0(self):
        """
        :obj: original Deep SORT model that was largely created from: https://github.com/ZQPei/deep_sort_pytorch/blob/master/deep_sort/deep/original_model.py
              input is expected to be of dim: [batch_size,3,128,64].
              output shape will be of dim: [batch_size, 128]
        """
        bb1 = BasicBlock(3,32,3,1,1,act=nn.ELU)
        bb2 = BasicBlock(32,32,3,1,1,act=nn.ELU)
        mp = nn.MaxPool2d(3,2,padding=1)
        conv = nn.Sequential(
            bb1,
            bb2,
            mp
        )
        # 32 64 32
        layer1 = rep_res_layers(32,32,2)
        # 32 64 32
        layer2 = rep_res_layers(32,64,2)
        # 64 32 16
        layer3 = rep_res_layers(64,128,2)
        # 128 16 8
        dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128*16*8, 128))

        self.net = nn.Sequential(conv, 
                                 layer1, layer2, layer3, 
                                 nn.Flatten(), dense, 
                                 nn.BatchNorm1d(128), Norm(p=2)) # input to `BatchNorm1d` MUST be > 1 as explained in: https://stackoverflow.com/questions/65882526/expected-more-than-1-value-per-channel-when-training-got-input-size-torch-size
    
    def init_archv1(self):
        """
        :obj: original model as presented in https://github.com/abhyantrika/nanonets_object_tracking/blob/master/siamese_net.py
              model is radically different than original deep SORT architecture
        """
        # define model parameters
        act: nn.modules.activation = self.act
        use_dropout: bool = self.use_dropout


        ops = nn.ModuleList()
        ops.append(BasicBlock(3,32,kernel_size=3,stride=2, act=act)) # shape: torch.Size([batch_size, 32, 127, 63])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(32,64,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 64, 63, 31])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(64,128,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 128, 31, 15])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(128,256,kernel_size=1,stride=2,act=act)) # shape: torch.Size([batch_size, 256, 16, 8])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,256,kernel_size=1,stride=2,act=act)) # shape: torch.Size([batch_size, 256, 8, 4])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,512,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 512, 3, 1])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        

        ops.append(BasicBlock(512,1024,kernel_size=(3,1),stride=1,act=act)) # shape: torch.Size([batch_size, 1024, 1, 1])
        ops.append(nn.Flatten()) # shape: torch.Size([batch_size, 1024])

        self.net = nn.Sequential(*ops)

