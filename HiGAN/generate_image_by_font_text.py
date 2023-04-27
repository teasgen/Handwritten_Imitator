import argparse
import yaml
import munch
from networks.model import AdversarialModel


def yaml2config(yml_path):
    with open(yml_path) as fp:
        json = yaml.load(fp, Loader=yaml.FullLoader)

    def to_munch(json):
        for key, val in json.items():
            if isinstance(val, dict):
                json[key] = to_munch(val)
        return munch.Munch(json)

    cfg = to_munch(json)
    return cfg


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
    cfg = yaml2config(args.config)
    model = AdversarialModel(cfg, args.config)
    model.load(args.ckpt, cfg.device)


def draw_font(font_example, text):
    return model.eval_text_custom_image(font_example, text)
