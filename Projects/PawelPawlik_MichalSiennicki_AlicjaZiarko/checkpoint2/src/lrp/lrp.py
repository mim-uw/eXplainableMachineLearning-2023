import torch
from model.resnet import get_resnet18
import hydra
import numpy as np
from torchvision.transforms.functional import to_pil_image
from datamodule.main import DataModule
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from zennit.composites import (
    EpsilonPlus,
    EpsilonAlpha2Beta1,
    EpsilonPlusFlat,
    EpsilonAlpha2Beta1Flat,
    EpsilonGammaBox,
    GuidedBackprop,
    ExcitationBackprop,
)
from zennit.torchvision import ResNetCanonizer
from zennit.attribution import Gradient
from zennit.image import imgify


N_EXAMPLES = 8
IN_CHANNELS = 3
OUT_FEATURES = 2
BATCH_SIZE = 1  # do not change!
FNT = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 17)
PATH = "epoch=5-step=2358.ckpt"  # "epoch=10-step=4323.ckpt"


def denormalize(img: torch.Tensor, mi=None, ma=None):
    mi = img.min() if mi is None else mi
    ma = img.min() if ma is None else ma
    return (img - mi) * 255 / (ma - mi)


@hydra.main(config_path="../conf/datamodule", config_name="melanoma", version_base=None)
def main(config) -> None:
    config.batch_size = BATCH_SIZE
    data_module: DataModule = hydra.utils.instantiate(config)
    dl = data_module.xai_dataloader()
    model = get_resnet18(IN_CHANNELS, OUT_FEATURES)
    model.load_state_dict({k[4:]: v for k, v in torch.load(PATH)["state_dict"].items()})
    model.eval()

    canonizers = [ResNetCanonizer()]

    # all rules: https://zennit.readthedocs.io/en/latest/how-to/use-rules-composites-and-canonizers.html
    composite_list = [
        EpsilonGammaBox(low=-1.0, high=1.0, canonizers=canonizers),
        EpsilonPlus(canonizers=canonizers),
        EpsilonPlusFlat(canonizers=canonizers),
        EpsilonAlpha2Beta1(canonizers=canonizers),
        EpsilonAlpha2Beta1Flat(canonizers=canonizers),
        GuidedBackprop(canonizers=canonizers),
        ExcitationBackprop(canonizers=canonizers),
    ]

    df = pd.DataFrame()
    for gt in [0.0, 1.0]:
        count = 0
        for x, y in dl:
            x: torch.Tensor
            assert x.shape[0] == 1
            if gt != y[0].item():
                continue
            if count > N_EXAMPLES // 2:
                break
            count += 1

            img_draw = to_pil_image(128 * (x[0] + 1))
            ImageDraw.Draw(img_draw).text((0, 0), f"y={y.item()}", (0, 0, 0), FNT)
            row = {"input": img_draw}

            for composite in composite_list:
                with Gradient(model=model, composite=composite) as attributor:
                    y_hat, relevance = attributor(x.float(), torch.eye(OUT_FEATURES)[y])
                    y_hat = torch.nn.Softmax(-1)(y_hat)

                # sum over the color channels
                heatmap = relevance.sum(1)
                # get the absolute maximum, to center the heat map around 0
                amax = heatmap.abs().numpy().max((1, 2))

                img_draw = imgify(
                    heatmap,
                    vmin=-amax,
                    vmax=amax,
                    cmap="coldnhot",
                    level=1.0,
                    grid=False,
                ).convert("RGB")
                ImageDraw.Draw(img_draw).text(
                    (0, 0), f"y_hat={y_hat[0, 1].item():.2f}", (255, 255, 0), FNT
                )
                row[composite.__class__.__name__] = img_draw

            df = pd.concat((df, pd.DataFrame([row])), ignore_index=True)

    df = df.transpose().reset_index()

    def foo(text):
        img = Image.new("RGB", x.shape[-2:], (217, 217, 217))
        ImageDraw.Draw(img).text((10, x.shape[-1] // 2), text, (0, 0, 0), FNT)
        return img

    df["index"] = df["index"].map(foo)

    df = df.applymap(np.array)
    arr = np.array(df.values.tolist())
    for _ in range(2):
        arr = np.concatenate(arr, axis=1)
    im = Image.fromarray(arr)
    im.save("lrp.png")


if __name__ == "__main__":
    main()
