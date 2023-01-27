import hydra
import torch
from hydra.utils import instantiate
from torch.nn import Linear, Conv2d
from torchvision.models import resnet18, resnet50


def get_resnet18(in_channels, out_features):
    res = resnet18(True)
    res.fc = Linear(512, out_features)
    res.conv1 = Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return res


def get_resnet50(in_channels, out_features):
    res = resnet50()
    res.fc = Linear(2048, out_features)
    res.conv1 = Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return res


@hydra.main(config_path="../conf/module", config_name="default", version_base=None)
def main(config):
    print(config)
    model = instantiate(config.model)
    print(model)

    print(model(torch.randn(1, 3, 256, 256)).shape)


if __name__ == "__main__":
    main()
