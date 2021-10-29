import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, load_state_dict_from_url
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from collections import OrderedDict
from typing import List, Union, Type
from torchvision.ops import MultiScaleRoIAlign

resnet_18_pretrained = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'


'''Represents the last three conv blocks in resnet. Used as the backbone for the FasterRCNN branches'''
class ResNetTail(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained=False
    ) -> None:
        super(ResNetTail, self).__init__(block, layers)

        if pretrained:
            state_dict = load_state_dict_from_url(resnet_18_pretrained, progress=True) #load pretrained imagenet resnet18 weights
            self.load_state_dict(state_dict)

        # remove unused layers from ResNet model
        self.conv1 = None
        self.bn1 = None
        self.maxpool = None
        self.avgpool = None
        self.layer1 = None
        self.relu = None
        self.fc = None
        self.out_channels = 512 

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



'''Represents a single FasterRCNN branch in the HydraNet. Backbone is ResNet-18. Implements the RPN.'''
class FasterRCNNBranch(FasterRCNN):
    def __init__(self, backbone, original_image_sizes, num_classes=9) -> None:
        box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

        super(FasterRCNNBranch, self).__init__(backbone=backbone, 
                                                num_classes=num_classes, 
                                                rpn_anchor_generator=anchor_generator, 
                                                box_roi_pool=box_roi_pool)
        self.original_image_sizes = original_image_sizes


    """
    Args:
        x (Tensor): partially processed features from images.
        images (ImageList): images to be processed
        targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    def forward(self, x, images, targets=None):
        original_sizes = [self.original_image_sizes for i in range(len(images.tensors))]
        features = self.backbone(x)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections



'''Forward facing camera branch implemented using Faster RCNN'''
class CameraBranch(FasterRCNNBranch):
    def __init__(self, backbone, transform, original_image_sizes=(376,672)) -> None:
        super(CameraBranch, self).__init__(backbone, original_image_sizes)
        self.transform = transform



'''Radar BEV branch implemented using Faster RCNN'''
class RadarBranch(FasterRCNNBranch):
    def __init__(self, backbone, transform, original_image_sizes=(576,1000)) -> None:
        super(RadarBranch, self).__init__(backbone, original_image_sizes)
        self.transform = transform



'''BEV lidar branch implemented using Faster RCNN'''
class LidarBranch(FasterRCNNBranch):
    def __init__(self, backbone, transform, original_image_sizes=(576,1000)) -> None:
        super(LidarBranch, self).__init__(backbone, original_image_sizes)
        self.transform = transform



'''Forward facing camera branch that concatenates L and R cameras via the channel dimension.'''
class DualCameraFusionBranch(FasterRCNNBranch):
    def __init__(self, backbone, transform, original_image_sizes=(376,672)) -> None:
        super(DualCameraFusionBranch, self).__init__(backbone, original_image_sizes)
        self.merge_conv = nn.Conv2d(128, 64, 3, 1, 1)
        self.transform = transform


    def forward(self, left_x, right_x, images, targets=None):
        x = torch.cat([left_x, right_x], dim=1) #concat over channel dim.
        x = self.merge_conv(x)
        return super(DualCameraFusionBranch, self).forward(x, images, targets)



'''Forward facing camera lidar fusion branch that combines L and R cameras and the forward lidar projection via the channel dimension.'''
class CameraLidarFusionBranch(FasterRCNNBranch):
    def __init__(self, backbone, transform, original_image_sizes=(376,672)) -> None:
        super(CameraLidarFusionBranch, self).__init__(backbone, original_image_sizes)
        self.merge_conv = nn.Conv2d(192, 64, 3, 1, 1)
        self.transform = transform


    def forward(self, left_x, right_x, r_lidar_x, images, targets=None):
        x = torch.cat([left_x, right_x, r_lidar_x], dim=1) #concat over channel dim.
        x = self.merge_conv(x)
        return super(CameraLidarFusionBranch, self).forward(x, images, targets)



'''BEV radar lidar fusion branch implemented using Faster RCNN'''
class RadarLidarFusionBranch(FasterRCNNBranch):
    def __init__(self, backbone, transform, original_image_sizes=(576,1000)) -> None:
        super(RadarLidarFusionBranch, self).__init__(backbone, original_image_sizes)
        self.merge_conv = nn.Conv2d(128, 64, 3, 1, 1)
        self.transform = transform


    def forward(self, radar_x, lidar_x, images, targets=None):
        x = torch.cat([radar_x, lidar_x], dim=1) #concat over channel dim.
        x = self.merge_conv(x)
        return super(RadarLidarFusionBranch, self).forward(x, images, targets)
