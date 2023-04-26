from copy import deepcopy

import cv2
import torch
from PIL import Image
from munch import Munch
from torchvision.transforms import Compose, Normalize, ToTensor
from networks.BigGAN_networks import Generator
from networks.module import StyleEncoder, StyleBackbone
from lib.alphabet import strLabelConverter, Alphabets


class BaseModel(object):
    def __init__(self, opt, log_root='./'):
        self.opt = opt
        self.local_rank = opt.local_rank if 'local_rank' in opt else -1
        self.device = torch.device(opt.device)
        self.models = Munch()
        self.models_ema = Munch()
        self.optimizers = Munch()
        self.log_root = log_root
        self.logger = None
        self.writer = None
        alphabet_key = 'rimes_word' if opt.dataset.startswith('rimes') else 'all'
        self.alphabet = Alphabets[alphabet_key]
        self.label_converter = strLabelConverter(alphabet_key)

    # def print(self, info):
    #     if self.logger is None:
    #         print(info)
    #     else:
    #         self.logger.info(info)
    #
    # def create_logger(self):
    #     if self.logger or self.writer:
    #         return
    #
    #     if not os.path.exists(self.log_root):
    #         os.makedirs(self.log_root)
    #
    #     self.writer = SummaryWriter(log_dir=self.log_root)
    #
    #     opt_str = option_to_string(self.opt)
    #     with open(os.path.join(self.log_root, 'config.txt'), 'w') as f:
    #         f.writelines(opt_str)
    #     print('log_root: ', self.log_root)
    #     self.logger = get_logger(self.log_root)
    #
    # def info(self, extra=None):
    #     self.print("RUNDIR: {}".format(self.log_root))
    #     opt_str = option_to_string(self.opt)
    #     self.print(opt_str)
    #     for model in self.models.values():
    #         self.print(_info(model, ret=True))
    #     if extra is not None:
    #         self.print(extra)
    #     self.print('=' * 20)
    #
    # def save(self, tag='best', epoch_done=0, **kwargs):
    #     ckpt = {}
    #     for model in self.models.values():
    #         ckpt[type(model).__name__] = model.state_dict()
    #
    #     for key, optim in self.optimizers.items():
    #         ckpt['OPT.' + key] = optim.state_dict()
    #
    #     for key, val in kwargs.items():
    #         ckpt[key] = val
    #
    #     ckpt['Epoch'] = epoch_done
    #     ckpt_save_path = os.path.join(self.log_root, self.opt.training.ckpt_dir, tag + '.pth')
    #     torch.save(ckpt, ckpt_save_path)

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

        for key in self.optimizers.keys():
            try:
                self.optimizers[key].load_state_dict(ckpt.pop('OPT.' + key))
            except Exception as e:
                print('Load %s failed' % ('OPT.' + key))

        ckpt['Epoch'] = 0 if 'Epoch' not in ckpt else ckpt['Epoch']
        return ckpt['Epoch']

    def set_mode(self, mode='eval'):
        for model in self.models.values():
            if mode == 'eval':
                model.eval()
            elif mode == 'train':
                model.train()
            else:
                raise NotImplementedError()

    # def validate(self, *args, **kwargs):
    #     yield NotImplementedError()
    #
    # def train(self):
    #     yield NotImplementedError()


class AdversarialModel(BaseModel):
    def __init__(self, opt, log_root='./'):
        super(AdversarialModel, self).__init__(opt, log_root)
        device = self.device
        generator = Generator(**opt.GenModel).to(device)
        style_backbone = StyleBackbone(**opt.StyBackbone).to(device)
        style_encoder = StyleEncoder(**opt.EncModel).to(device)

        self.models = Munch(
            G=generator,
            E=style_encoder,
            B=style_backbone,
        )

    def eval_text_custom_image(self, org_img, text):
        self.set_mode('eval')

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
            print(f'Generator has {sum(p.numel() for p in self.models.G.parameters())} parameters')
            print(f'Encoder has {sum(p.numel() for p in self.models.E.parameters())} parameters')
            gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
            return gen_imgs
