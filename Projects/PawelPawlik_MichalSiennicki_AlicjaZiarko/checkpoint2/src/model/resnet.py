import hydra
import torch
from hydra.utils import instantiate
from torch.nn import Linear, Conv2d
from torchvision.models import resnet18


def get_resnet18(in_channels, out_features):
    res = resnet18(True)
    res.fc = Linear(512, out_features)
    res.conv1 = Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return res


@hydra.main(config_path="../conf/model", config_name="resnet18", version_base=None)
def main(config):
    print(config)
    model = instantiate(config)
    print(model)

    print(model(torch.randn(1, 1, 256, 256)).shape)


if __name__ == "__main__":
    main()
