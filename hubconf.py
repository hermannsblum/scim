import torch
from torchvision.models.segmentation import deeplabv3_resnet101

dependencies = ["torch", "torchvision"]


def dv3res101_no_tv(pretrained=True, **kwargs):
    """
    Deeplabv3+ with ResNet101 trained on COCO and ScanNet without TVs.
    Output classes are NYU40.
    """
    model = deeplabv3_resnet101(pretrained=False, num_classes=40, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/6840795/files/deeplab_no_tv.pth?download=1",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def dv3res101_no_book(pretrained=True, **kwargs):
    """
    Deeplabv3+ with ResNet101 trained on COCO and ScanNet without books.
    Output classes are NYU40.
    """
    model = deeplabv3_resnet101(pretrained=False, num_classes=40, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/6840795/files/deeplab_no_books.pth?download=1",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def dv3res101_no_towel(pretrained=True, **kwargs):
    """
    Deeplabv3+ with ResNet101 trained on COCO and ScanNet without towels.
    Output classes are NYU40.
    """
    model = deeplabv3_resnet101(pretrained=False, num_classes=40, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/6840795/files/deeplab_no_towel.pth?download=1",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
    return model
