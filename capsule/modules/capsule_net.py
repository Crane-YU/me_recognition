import torch.nn as nn
from .capsule_layers import PrimaryCapsule, MECapsule
from torchvision import models
import torch.nn.functional as F


class ResNetLayers(nn.Module):
    def __init__(self, is_freeze=False):
        super(ResNetLayers, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # Delete the following layers
        delattr(self.model, 'layer4')
        delattr(self.model, 'avgpool')
        delattr(self.model, 'fc')

        if is_freeze:
            for index, p in enumerate(self.model.parameters()):
                if index == 15:
                    break
                p.requires_grad = False

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        return output


class VGGLayers(nn.Module):
    def __init__(self, is_freeze=True):
        super(VGGLayers, self).__init__()
        self.model = models.vgg11(pretrained=True).features[:11]

        if is_freeze:
            for i in range(4):
                for p in self.model[i].parameters():
                    p.requires_grad = False

    def forward(self, x):
        # input : [bs, 3, 224, 224]
        return self.model(x)  # output : [bs, 256, 20, 20]


backbone = {'vggNet': VGGLayers, 'resNet': ResNetLayers}


class MECapsuleNet(nn.Module):
    """
    A Capsule Network on Micro-expression.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param num_iter: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """

    def __init__(self, input_size, classes, num_iter, model_name='resNet', is_freeze=True):
        super(MECapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.num_iter = num_iter

        self.conv = backbone[model_name](is_freeze)

        self.conv1 = nn.Conv2d(256, 256, kernel_size=9, stride=1, padding=0)

        self.primaryCaps = PrimaryCapsule(256, 32 * 8, 8, kernel_size=9, stride=2, padding=0)

        self.digitCaps = MECapsule(in_num_caps=32 * 6 * 6,
                                   in_dim_caps=8,
                                   out_num_caps=self.classes,
                                   out_dim_caps=16,
                                   num_iter=num_iter)

    def forward(self, x, y=None):
        x = self.conv(x)
        x = F.relu(self.conv1(x))
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        length = x.norm(dim=-1)
        return length
