import csv
from os import path

import PIL as pil
import torch
from torch.nn import functional as nnf
from torch.utils import data as tdata
from torch import distributions as tdist
from torchvision.transforms import functional as ttf

def join_via_addition(img, xx, yy, probs):
    img[yy, xx] += probs

def join_via_replacement(img, xx, yy, probs):
    img[yy, xx] = probs

def join_via_max(img, xx, yy, probs):
    img[yy, xx] = torch.max(img[yy, xx], probs)

def generate_via_multivariate_normal(peak = 1, variance_func=lambda a: (a/2)**2):
    def do_generate(cx, cy, width, height, xx, yy):
        distr = tdist.MultivariateNormal(
            torch.tensor([cy, cx], dtype=torch.float),
            torch.tensor([[variance_func(height), 0], [0, variance_func(width)]], dtype=torch.float)
        )
        log_probs = distr.log_prob(torch.dstack((yy, xx)))
        probs = torch.exp(log_probs)
        normalized_probs = probs / torch.max(probs) * peak
        return normalized_probs
    return do_generate

def generate_via_kant_method(size=120, sigma=40):
    cx = size // 2
    cy = size // 2
    coord_range = torch.arange(size)

    xx, yy = torch.meshgrid((coord_range, coord_range))
    xx = (xx - cx)**2
    yy = (yy - cy)**2

    base = -4 * torch.log(torch.tensor(2.0)) * (xx + yy) / sigma ** 2
    base = torch.exp(base)

    base = base[None, None] # introduce minibatch- and channel dims for nnf.interpolate
    def do_generate(cx, cy, width, height, xx, yy):
        return nnf.interpolate(base, size=(xx.shape[0], yy.shape[1]), mode='bilinear', align_corners=False)
    return do_generate

def generate_gaussians(w, h, boxes, size_reduction=1, generate_method=generate_via_multivariate_normal(), join_method=join_via_max):
    w = w // size_reduction
    h = h // size_reduction
    img = torch.zeros((h, w))

    for b in boxes:
        x1, y1, x2, y2 = b // size_reduction
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = torch.abs(x2 - x1)
        height = torch.abs(y2 - y1)

        x_range = torch.arange(max(x1, 0), min(x2, w), dtype=torch.float)
        y_range = torch.arange(max(y1, 0), min(y2, h), dtype=torch.float)
        xx, yy = torch.meshgrid((x_range, y_range))

        probs = generate_method(cx, cy, width, height, xx, yy)

        join_method(img, xx.long(), yy.long(), probs)

    return img

def sku110k_flip(image, targets, gaussians = True):
    image = ttf.hflip(image)

    w = image.shape[-1] if torch.is_tensor(image) else image.width
    flipped_boxes = targets['boxes'].clone()
    flipped_boxes[:, 0] = w - targets['boxes'][:, 2]
    flipped_boxes[:, 2] = w - targets['boxes'][:, 0]
    targets['boxes'] = flipped_boxes

    if gaussians:
        targets['gaussians'] = ttf.hflip(targets['gaussians'])

    return image, targets

def sku110k_collate_fn(samples):
    return SKU110KBatch(samples)

def sku110k_no_gauss_collate_fn(samples):
    return SKU110KBatch(samples, gaussians=False)

class SKU110KBatch:
    def __init__(self, samples, gaussians=True):
        self.images, self.targets = zip(*samples)
        self.tensor_target_keys = ['boxes', 'labels', 'gaussians'] if gaussians else ['boxes', 'labels']
    def __getitem__(self, i):
        if i == 0: return self.images
        if i == 1: return self.targets
        raise IndexError
    def pin_memory(self):
        for i in self.images:
            i.pin_memory()

        for t in self.targets:
            for g in self.tensor_target_keys:
                if g not in t: continue
                t[g].pin_memory()

        return self
    def cuda(self, device=None, non_blocking=True):
        self.images = [i.cuda(device, non_blocking=non_blocking) for i in self.images]
        self.targets = [{k: t[k].cuda(device, non_blocking=non_blocking) for k in self.tensor_target_keys} for t in self.targets]
        return self

class SKU110KDataset(tdata.Dataset):
    def __init__(self, img_dir_path, annotation_file_path, skip=[],
    include_gaussians=True, gauss_generate_method=generate_via_multivariate_normal, gauss_join_method=join_via_max,
    flip_chance=0.5): # TODO: more augmentation stuff
        super().__init__()
        self.img_dir = img_dir_path
        self.index = self.build_index(annotation_file_path, skip)
        self.include_gaussians = include_gaussians
        self.generate_method = gauss_generate_method
        self.join_method = gauss_join_method
        self.flip_chance = flip_chance
    def build_index(self, annotation_file_path, skip):
        index = {}
        print('Building index...')
        with open(annotation_file_path, 'r') as annotation_file:
            annotation_reader = csv.reader(annotation_file)
            for row in annotation_reader:
                if len(row) != 8:
                    print(f'Malformed annotation row: {row}, skipping')
                    continue
                name, x1, y1, x2, y2, _, img_width, img_height = row
                if name in skip:
                    continue
                if name not in index:
                    index[name] = {'image_name': name, 'image_width': int(img_width), 'image_height': int(img_height), 'boxes': []}
                index[name]['boxes'].append(torch.tensor([int(coord) for coord in (x1, y1, x2, y2)]))
        print('Finishing up...')
        for val in index.values():
            val['labels'] = torch.zeros(len(val['boxes']), dtype=torch.long)
            val['boxes'] = torch.stack(val['boxes'])
        res = list(index.values())
        print('Done!')
        return res
    def index_for_name(self, name):
        for i, entry in enumerate(self.index):
            if entry['image_name'] == name:
                return i
        return None
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        index_entry = {**self.index[i]} # copy the dict so that gaussians dont get stored in the index
        img_path = path.join(self.img_dir, index_entry['image_name'])
        img = pil.Image.open(img_path)
        if self.include_gaussians:
            index_entry['gaussians'] = generate_gaussians(
                index_entry['image_width'], index_entry['image_height'], index_entry['boxes'],
                generate_method=self.generate_method(), join_method=self.join_method
            )
        if torch.rand(1) <= self.flip_chance:
            img, index_entry = sku110k_flip(img, index_entry, self.include_gaussians)
        try:
            return ttf.to_tensor(img), index_entry
        except OSError:
            print(f'WARNING: Malformed image: {index_entry["image_name"]}'
                + f' - You\'ll probably want to explicitly skip this! Returning image 0 ({self.index[0]["image_name"]}) instead.')
            return self[0]
