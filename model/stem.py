from local_torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, load_state_dict_from_url
from typing import Type, Union, List

'''Represents the Stem block in the HydraNet. Backbone is ResNet-18'''

resnet_18_pretrained = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'


class Stem(ResNet):
    def __init__(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained=False,
    ) -> None:
        super(Stem, self).__init__(block, layers)

        if pretrained:
            state_dict = load_state_dict_from_url(resnet_18_pretrained, progress=True) #load pretrained imagenet resnet18 weights
            self.load_state_dict(state_dict)
        
        # remove unused layers from ResNet model
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.fc = None
        self.avgpool = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x 


class CameraStem(Stem):
    def __init__(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained=False
    ) -> None:
        super(CameraStem, self).__init__(block, layers, pretrained)


class RadarStem(Stem):
    def __init__(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained=False
    ) -> None:
        super(RadarStem, self).__init__(block, layers, pretrained)
        

class LidarStem(Stem):
    def __init__(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained=False
    ) -> None:
        super(LidarStem, self).__init__(block, layers, pretrained)
        