import torch
import torch.nn as nn
from torchvision.models import resnet18


class MyResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = nn.CrossEntropyLoss()

        ############################################################################
        # Student code begin
        ############################################################################

        self.num_classes = 15
        pretrained = resnet18(pretrained=True)
        conv_layers = []
        for c in pretrained.children():
            if not isinstance(c,nn.Linear):
                for param in c.parameters():
                    param.requires_grad = False
                conv_layers.append(c)
        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc_layers = nn.Linear(in_features=512, out_features=self.num_classes)

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`my_resnet.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        ############################################################################
        # Student code begin
        ############################################################################
    
        N = x.shape[0]
        x = self.conv_layers(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        model_output = torch.reshape(self.fc_layers(x), (N,self.num_classes))

        # raise NotImplementedError(
        #     "`forward` function in "
        #     + "`my_resnet.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################
        return model_output
