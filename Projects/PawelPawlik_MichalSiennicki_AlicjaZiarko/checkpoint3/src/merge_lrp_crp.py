import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=Path, default=Path("ckpt"))
    parser.add_argument("--crp_rows", type=int, default=3)
    return parser.parse_args()


def main(args) -> None:
    for crp_path in args.dir.glob("crp_*.png"):
        crp_path: Path
        lrp_path = crp_path.with_name("short_" + crp_path.name[4:])
        assert crp_path.exists() and lrp_path.exists(), str(crp_path, lrp_path)

        crp_img = imageio.imread(crp_path)
        lrp_img = imageio.imread(lrp_path)

        box_size = lrp_img.shape[0] // 3

        res_img = np.concatenate([lrp_img, crp_img[-args.crp_rows * box_size :]])
        res_path = crp_path.with_name("lrp_" + crp_path.name)
        print(lrp_img.shape, res_img.shape)
        Image.fromarray(res_img[box_size // 3:]).save(res_path)


if __name__ == "__main__":
    main(parse_args())
