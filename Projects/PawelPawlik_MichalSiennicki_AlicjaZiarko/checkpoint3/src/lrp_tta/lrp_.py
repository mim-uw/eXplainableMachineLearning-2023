from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from zennit.attribution import Gradient
from zennit.composites import (
    EpsilonPlus,
    EpsilonAlpha2Beta1,
    EpsilonPlusFlat,
    EpsilonAlpha2Beta1Flat,
    EpsilonGammaBox,
    GuidedBackprop,
    ExcitationBackprop,
)
from zennit.image import imgify
from zennit.torchvision import ResNetCanonizer, VGGCanonizer
from itertools import product
import argparse
from lrp_tta.config import *
from torchvision.transforms import transforms

from datamodule.main import DataModule

BATCH_SIZE = 1  # do not change!
FNT = ImageFont.truetype("DejaVuSansMono.ttf", 30)
SPACING = 50
BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
TRANSFORMS = [
    ("identity", transforms.RandomRotation(0)),
    ("blur", transforms.GaussianBlur(5)),
    # ("erasing", transforms.RandomErasing(p=1.0)),
    ("sharpen", transforms.RandomAdjustSharpness(2, p=1.0)),
    # (
    #     "crop",
    #     transforms.Compose(
    #         [transforms.CenterCrop(150), transforms.Resize(190)],
    #     ),
    # ),
    ("rotate90", transforms.RandomRotation((90, 90))),
    ("rotate15", transforms.RandomRotation((15, 15))),
]
COVID_TRANSFORMS = {
    "rotate90": ("rotate-10", transforms.RandomRotation((-10, -10))),
    "rotate15": ("rotate10", transforms.RandomRotation((10, 10))),
}


def denormalize(img: torch.Tensor, mi=None, ma=None):
    mi = img.min() if mi is None else mi
    ma = img.min() if ma is None else ma
    return (img - mi) * 255 / (ma - mi)


def get_config(config_path_with_name, overrides):
    config_path_with_name = Path(config_path_with_name)
    if not config_path_with_name.is_absolute():
        config_path_with_name = Path("..") / config_path_with_name
    with hydra.initialize(config_path=str(config_path_with_name.parent)):
        return hydra.compose(
            config_name=str(config_path_with_name.name), overrides=overrides
        )


def main(args) -> None:
    config = get_config(
        args.config_path, [f"module={args.arch}"] if args.arch is not None else []
    )
    cls = COVID_CLS if config.module.num_classes == 3 else MELANOMA_CLS
    samples = None
    if args.fixed_samples:
        samples = COVID_SAMPLES if config.module.num_classes == 3 else MELANOMA_SAMPLES
    config.datamodule.batch_size = BATCH_SIZE
    data_module: DataModule = hydra.utils.instantiate(config.datamodule)
    dl = data_module.xai_dataloader()
    model = hydra.utils.instantiate(config.module.model)
    model.load_state_dict(
        {
            k[4:]: v
            for k, v in torch.load(args.model_path)["state_dict"].items()
            if k[:4] == "net."
        }
    )
    model.eval()

    if args.arch == "vgg":
        canonizers = [VGGCanonizer()]
    else:
        canonizers = [ResNetCanonizer()]

    if args.short:
        composite_list = [EpsilonPlusFlat(canonizers=canonizers)]
    else:
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
    for gt, pred in product(
        range(config.module.num_classes), [0, 1] if samples is None else [None]
    ):
        count = 0
        for idx, (_x, y) in enumerate(dl):
            _x: torch.Tensor
            assert _x.shape[0] == 1
            if samples is None:
                if count >= args.n_examples:
                    break
                if gt != y[0].item():
                    continue
                y_hat = torch.nn.Softmax(-1)(model(_x))[0, gt].item()
                if pred == 1 and y_hat < 0.7:
                    continue
                if pred == 0 and y_hat > 0.3:
                    continue
            else:
                if count >= len(samples[gt]):
                    break
                if idx not in samples[gt]:
                    continue
                y_hat = torch.nn.Softmax(-1)(model(_x))[0, gt].item()
            count += 1

            for t_name, t in TRANSFORMS:
                if "covid" in args.model_path.parts and t_name in COVID_TRANSFORMS:
                    t_name, t = COVID_TRANSFORMS[t_name]

                x = _x.clone().detach()
                x = t(x)

                y_hat = torch.nn.Softmax(-1)(model(x))[0, gt].item()

                img_draw = Image.new("RGB", x.shape[-2:], BACKGROUND_COLOR)
                ImageDraw.Draw(img_draw).text(
                    (10, x.shape[-1] - 3 * SPACING), t_name, TEXT_COLOR, FNT
                )
                ImageDraw.Draw(img_draw).text(
                    (10, x.shape[-1] - 2 * SPACING), f"{cls[y.item()]}", TEXT_COLOR, FNT
                )
                ImageDraw.Draw(img_draw).text(
                    (10, x.shape[-1] - SPACING), f"{y_hat:.2f}", TEXT_COLOR, FNT
                )
                # ImageDraw.Draw(img_draw).text(
                #     (10, x.shape[-1] - 4 * SPACING), f"id: {idx}", TEXT_COLOR, FNT
                # )

                row = {
                    "": img_draw,
                    "sample": (x[0].permute(1, 2, 0) * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8),
                }

                for composite in composite_list:
                    with Gradient(model=model, composite=composite) as attributor:
                        _, relevance = attributor(
                            x.float(), torch.eye(config.module.model.out_features)[y]
                        )

                    # sum over the color channels
                    heatmap = relevance.sum(1)
                    # get the absolute maximum, to center the heat map around 0
                    amax = heatmap.abs().numpy().max((1, 2))

                    row[composite.__class__.__name__] = imgify(
                        heatmap,
                        vmin=-amax,
                        vmax=amax,
                        cmap="coldnhot",
                        level=1.0,
                        grid=False,
                    ).convert("RGB")

                df = pd.concat((df, pd.DataFrame([row])), ignore_index=True)

    df = df.transpose().reset_index()

    def foo(text):
        img = Image.new("RGB", x.shape[-2:], BACKGROUND_COLOR)
        ImageDraw.Draw(img).text((10, x.shape[-1] // 2), text, TEXT_COLOR, FNT)
        return img

    df["index"] = df["index"].map(foo)

    if args.skip_row_name:
        df = df.drop(columns=["index"])

    df = df.applymap(np.array)
    arr = np.array(df.values.tolist())
    for _ in range(2):
        arr = np.concatenate(arr, axis=1)
    im = Image.fromarray(arr)
    print(args.model_path.with_suffix(".png"))
    name = f"tta_{args.model_path.parts[-3]}_{args.model_path.parts[-2]}.png"
    if args.short:
        name = "short_" + name

    im.save(args.output_path if args.output_path is not None else Path("ckpt") / name)
    im.save(
        args.output_path
        if args.output_path is not None
        else args.model_path.with_name(name)
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=Path, required=True)
    parser.add_argument("-c", "--config_path", type=Path, default=Path("conf/main"))
    parser.add_argument("-n", "--n_examples", type=int, default=2)
    parser.add_argument("--fixed_samples", action="store_true")
    parser.add_argument("-o", "--output_path", type=Path)
    parser.add_argument("-a", "--arch", type=str)
    parser.add_argument("--skip_row_name", action="store_true")
    parser.add_argument("--short", action="store_true")
    return parser


if __name__ == "__main__":
    main(parser().parse_args())
