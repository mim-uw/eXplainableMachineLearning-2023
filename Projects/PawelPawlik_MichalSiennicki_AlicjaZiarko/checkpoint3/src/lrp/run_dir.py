from pathlib import Path
from lrp.lrp_ import main as run_lrp, parser
from tqdm import tqdm
import argparse


def main(_args) -> None:
    tqdm_bar = tqdm(list(_args.dir.rglob("*.ckpt")))
    for path in tqdm_bar:
        tqdm_bar.set_description(str(path))

        config = "conf/main"
        if "covid" in path.parts:
            config = "conf/covid"

        args = [
            f"--model_path={path}",
            f"--config_path={config}",
            "--fixed_samples",
            "--skip_row_name",
        ]
        if "vgg" in path.parts:
            args.append("--arch=vgg")
        if _args.short:
            args.append("--short")

        run_lrp(parser().parse_args(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=Path, default=Path("ckpt"))
    parser.add_argument("--short", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
