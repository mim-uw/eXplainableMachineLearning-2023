import hydra
import torch
from hydra.utils import instantiate
from torch.nn import Linear, Conv2d
from torchvision.models import vgg16


def get_vgg16(in_channels, out_features):
    res = vgg16(True)
    res.classifier[-1] = Linear(4096, out_features)
    assert in_channels == 3
    return res


@hydra.main(config_path="../conf/module", config_name="vgg", version_base=None)
def main(config):
    print(config)
    model = instantiate(config.model)
    print(model)

    print(model(torch.randn(1, 3, 256, 256)).shape)


if __name__ == "__main__":
    main()
