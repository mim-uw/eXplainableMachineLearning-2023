import hydra
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from zennit.attribution import Gradient

from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm

from zennit.image import imgify
from zennit.torchvision import ResNetCanonizer

from datamodule.main import DataModule

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.image import imgify

from crp.helper import get_layer_names






N_EXAMPLES = 8
BATCH_SIZE = 1  # do not change!
FNT = ImageFont.truetype("DejaVuSansMono.ttf", 17)
PATH = "/home/alicja/studia/xai_crp_lrp/good_model.ckpt"


def denormalize(img: torch.Tensor, mi=None, ma=None):
    mi = img.min() if mi is None else mi
    ma = img.min() if ma is None else ma
    return (img - mi) * 255 / (ma - mi)


@hydra.main(config_path="../conf", config_name="main", version_base=None)
def main(config) -> None:
    config.datamodule.batch_size = BATCH_SIZE
    data_module: DataModule = hydra.utils.instantiate(config.datamodule)
    dl = data_module.xai_dataloader()
    model = hydra.utils.instantiate(config.module.model)
    model.load_state_dict({k[4:]: v for k, v in torch.load(PATH)["state_dict"].items() if k[:4] == "net."})
    model.eval()

    cc = ChannelConcept()
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    attribution = CondAttribution(model)

    # compute heatmap wrt. output 46 (green lizard class)
    conditions = [{"y": 1}]

    for i, (x, y) in enumerate(dl):
        if y[0].item() != 1.0:
            continue
        sample = x
        # zennit requires gradients
        sample.requires_grad = True
        attr = attribution(sample, conditions, composite, mask_map=cc.mask)

        # or use a dictionary for mask_map
        layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        mask_map = {name: cc.mask for name in layer_names}

        attr = attribution(sample, conditions, composite, mask_map=mask_map)



        print(torch.equal(attr[0], attr.heatmap))

        img1 = imgify(attr.heatmap, symmetric=True).convert("RGB")
        img2 = imgify(x[0], symmetric=True).convert("RGB")
        img2.save(f"crp{i}.png")
        img1.save(f"attr{i}.png")
        if i > 10:
            break


    # canonizers = [ResNetCanonizer()]

    # # all rules: https://zennit.readthedocs.io/en/latest/how-to/use-rules-composites-and-canonizers.html
    # composite_list = [
    #     EpsilonGammaBox(low=-1.0, high=1.0, canonizers=canonizers),
    #     EpsilonPlus(canonizers=canonizers),
    #     EpsilonPlusFlat(canonizers=canonizers),
    #     EpsilonAlpha2Beta1(canonizers=canonizers),
    #     EpsilonAlpha2Beta1Flat(canonizers=canonizers),
    #     GuidedBackprop(canonizers=canonizers),
    #     ExcitationBackprop(canonizers=canonizers),
    # ]

    # df = pd.DataFrame()
    # for gt in [0.0, 1.0]:
    #     count = 0
    #     for x, y in dl:
    #         x: torch.Tensor
    #         assert x.shape[0] == 1
    #         if gt != y[0].item():
    #             continue
    #         if count > N_EXAMPLES // 2:
    #             break
    #         count += 1

    #         img_draw = to_pil_image(x[0])
    #         ImageDraw.Draw(img_draw).text((0, 0), f"y={y.item()}", (0, 0, 0), FNT)
    #         row = {"input": img_draw}

    #         for composite in composite_list:
    #             with Gradient(model=model, composite=composite) as attributor:
    #                 y_hat, relevance = attributor(x.float(), torch.eye(config.module.model.out_features)[y])
    #                 y_hat = torch.nn.Softmax(-1)(y_hat)

    #             # sum over the color channels
    #             heatmap = relevance.sum(1)
    #             # get the absolute maximum, to center the heat map around 0
    #             amax = heatmap.abs().numpy().max((1, 2))

    #             img_draw = imgify(
    #                 heatmap,
    #                 vmin=-amax,
    #                 vmax=amax,
    #                 cmap="coldnhot",
    #                 level=1.0,
    #                 grid=False,
    #             ).convert("RGB")
    #             ImageDraw.Draw(img_draw).text(
    #                 (0, 0), f"y_hat={y_hat[0, 1].item():.2f}", (255, 255, 0), FNT
    #             )
    #             row[composite.__class__.__name__] = img_draw

    #         df = pd.concat((df, pd.DataFrame([row])), ignore_index=True)

    # df = df.transpose().reset_index()

    # def foo(text):
    #     img = Image.new("RGB", x.shape[-2:], (217, 217, 217))
    #     ImageDraw.Draw(img).text((10, x.shape[-1] // 2), text, (0, 0, 0), FNT)
    #     return img

    # df["index"] = df["index"].map(foo)

    # df = df.applymap(np.array)
    # arr = np.array(df.values.tolist())
    # for _ in range(2):
    #     arr = np.concatenate(arr, axis=1)
    # im = Image.fromarray(arr)
    # im.save("crp.png")


if __name__ == "__main__":
    main()
