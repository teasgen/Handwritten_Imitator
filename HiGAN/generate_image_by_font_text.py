import argparse
from networks.model import AdversarialModel

model = None


def init_model():
    global model
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        default="configs/gan_iam.yml",
    )

    parser.add_argument(
        "--ckpt",
        default="./pretrained/deploy_HiGAN+.pth",
    )

    parser.add_argument(
        "--mode",
        default="text",
    )

    args = parser.parse_args()
    model = AdversarialModel(args.config)
    model.load(args.ckpt, 'cuda:0')


def draw_font(font_example, text):
    return model.eval_text_custom_image(font_example, text)
