from copy import deepcopy

import cv2
import torch
from PIL import Image
from munch import Munch
from torchvision.transforms import Compose, Normalize, ToTensor
from networks.BigGAN_networks import Generator
from networks.module import StyleEncoder, StyleBackbone
from lib.alphabet import strLabelConverter, Alphabets


class Model(object):
    def __init__(self):
        self.device = torch.device('cuda:0')
        alphabet_key = 'all'
        self.alphabet = Alphabets[alphabet_key]
        self.label_converter = strLabelConverter(alphabet_key)
        generator = Generator().to(self.device)
        style_backbone = StyleBackbone().to(self.device)
        style_encoder = StyleEncoder().to(self.device)

        self.models = Munch(
            G=generator,
            E=style_encoder,
            B=style_backbone,
        )

    def load(self, ckpt, map_location=None, modules=None):
        if modules is None:
            modules = []
        elif not isinstance(modules, list):
            modules = [modules]

        print('load checkpoint from ', ckpt)
        if map_location is None:
            ckpt = torch.load(ckpt)
        else:
            ckpt = torch.load(ckpt, map_location=map_location)

        if ckpt is None:
            return

        models = self.models.values() if len(modules) == 0 else modules
        for model in models:
            try:
                model.load_state_dict(ckpt.pop(type(model).__name__))
            except Exception as e:
                print('Load %s failed' % type(model).__name__)

        ckpt['Epoch'] = 0 if 'Epoch' not in ckpt else ckpt['Epoch']
        return ckpt['Epoch']

    def eval_text_custom_image(self, org_img, text):
        for model in self.models.values():
            model.eval()

        def get_style_img(img):
            transf = Compose([ToTensor(), Normalize([0.5], [0.5])])
            img = img.squeeze()
            h, w = img.shape[:2]
            new_w = w * 64 // h
            dim = (new_w, 64)

            if new_w < w:
                style_img = cv2.resize(deepcopy(img), dim, interpolation=cv2.INTER_AREA)
            else:
                style_img = cv2.resize(deepcopy(img), dim, interpolation=cv2.INTER_LINEAR)
            style_img = Image.fromarray(style_img, mode='L')

            style_img = transf(deepcopy(style_img))
            return style_img.unsqueeze(0).to(self.device), torch.LongTensor([style_img.shape[-1]]).to(self.device)

        with torch.no_grad():
            real_img, real_img_len = get_style_img(org_img)

            fake_lbs = self.label_converter.encode(text)
            fake_lbs = torch.LongTensor(fake_lbs)
            fake_lb_lens = torch.IntTensor([fake_lbs.shape[0]])

            nrow = 1
            fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
            fake_lb_lens = fake_lb_lens.repeat(nrow, ).to(self.device)
            fake_lbs = fake_lbs.repeat(nrow, 1)
            fake_lb_lens = fake_lb_lens.repeat(nrow, )
            enc_styles = self.models.E(real_img, real_img_len, self.models.B)

            gen_imgs = self.models.G(enc_styles, fake_lbs, fake_lb_lens)
            gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
            return gen_imgs