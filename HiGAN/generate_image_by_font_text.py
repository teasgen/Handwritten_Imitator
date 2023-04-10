import argparse

import numpy as np
from PIL import Image

from lib.utils import yaml2config
from networks import get_model

model = None


def init_model():
    global model
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/gan_iam.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--ckpt",
        nargs="?",
        type=str,
        default="./pretrained/deploy_HiGAN+.pth",
        help="checkpoint for evaluation",
    )

    parser.add_argument(
        "--mode",
        nargs="?",
        type=str,
        default="text",
        help="mode: [rand] [style] [text] [interp]",
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)

    model = get_model(cfg.model)(cfg, args.config)
    model.load(args.ckpt, cfg.device)


def draw_font(font_example, text):
    # image = np.asarray(Image.open("data/background_removed.png"))
    # print(image.shape)
    # model.eval_text_custom_image(image, "Vlad sucks")
    return model.eval_text_custom_image(font_example, text)


# draw_font(None, None)
