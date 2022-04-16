import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule



class ReID_Architectures(LightningModule):
    def init_archv1(self):
        
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

        self.net = nn.Sequential(*ops)

    def init_archv2(self):
        """
        version is identical to v1 but we replaced each basic block with a residual block for every operation that contains
        `kernel_size=1`
        """
        
        # define model parameters
        act: nn.modules.activation = self.act
        use_dropout: bool = self.use_dropout


        ops = nn.ModuleList()
        ops.append(BasicBlock(3,32,kernel_size=3,stride=2, act=act)) # shape: torch.Size([batch_size, 32, 127, 63])
        ops.append(ResidualBlock(32,32))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(32,64,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 64, 63, 31])
        ops.append(ResidualBlock(64,64))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(64,128,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 128, 31, 15])
        ops.append(ResidualBlock(128,128))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(128,256,kernel_size=1,stride=2,act=act)) # shape: torch.Size([batch_size, 256, 16, 8])
        ops.append(ResidualBlock(256,256))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,256,kernel_size=1,stride=2,act=act)) # shape: torch.Size([batch_size, 256, 8, 4])
        ops.append(ResidualBlock(256,256))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,512,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 512, 3, 1])
        ops.append(ResidualBlock(512,512))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        

        ops.append(BasicBlock(512,1024,kernel_size=(3,1),stride=1,act=act)) # shape: torch.Size([batch_size, 1024, 1, 1])

        self.net = nn.Sequential(*ops)


    def init_archv3(self):
        """
        version is identical to v1 but we replaced each basic block with a residual block for every operation that contains
        `kernel_size=1`
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
        ops.append(ResidualBlock(128,256))
        ops.append(nn.MaxPool2d(1,stride=2,padding=0)) # shape: torch.Size([batch_size, 256, 16, 8])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(ResidualBlock(256,256))
        ops.append(nn.MaxPool2d(1,stride=2,padding=0)) # shape: torch.Size([batch_size, 256, 8, 4])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,512,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 512, 3, 1])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        

        ops.append(BasicBlock(512,1024,kernel_size=(3,1),stride=1,act=act)) # shape: torch.Size([batch_size, 1024, 1, 1])

        self.net = nn.Sequential(*ops)