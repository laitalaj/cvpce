import csv, json, os, re
from os import path

import PIL as pil
import torch
from torch.nn import functional as nnf
from torch.utils import data as tdata
from torch import distributions as tdist
from torchvision.transforms import functional as ttf
from torchvision import utils as tvutils

from . import planogram_adapters, utils

## PROPOSALS ##

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
    flip_chance=0.5): # TODO: more augmentation stuff, maybe?
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
        max_annotations = 0
        for val in index.values():
            max_annotations = max(len(val['boxes']), max_annotations)
            val['labels'] = torch.zeros(len(val['boxes']), dtype=torch.long)
            val['boxes'] = torch.stack(val['boxes'])
        res = list(index.values())
        print(f'Done! (max annotations in one image: {max_annotations})')
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

## CLASSIFICATION ##

CLASSIFICATION_IMAGE_SIZE = 256

def resize_for_classification(img):
    _, h, w = img.shape
    larger_dim = max(w, h)
    res = torch.full((3, larger_dim, larger_dim), 0.5)
    res[:, 0:h, 0:w] = img
    return ttf.resize(res, (CLASSIFICATION_IMAGE_SIZE, CLASSIFICATION_IMAGE_SIZE))

class TargetDomainDataset(SKU110KDataset):
    def __init__(self, img_dir_path, annotation_file_path, skip=[]):
        super().__init__(img_dir_path, annotation_file_path, skip, include_gaussians=False, flip_chance=0)
        self.bbox_index = self.build_bbox_index()
    def build_bbox_index(self):
        bbox_counts = torch.empty(len(self.index), dtype=torch.long)
        for i, entry in enumerate(self.index):
            bbox_counts[i] = len(entry['boxes'])
        return bbox_counts.cumsum(0)
    def __len__(self):
        return self.bbox_index[-1].item()
    def __getitem__(self, i):
        image_idx = torch.nonzero(self.bbox_index > i)[0, 0] # 0, 0 for the first nonzero (=true) index
        bbox_idx = i - self.bbox_index[image_idx - 1] if image_idx > 0 else i

        img, index_entry = super().__getitem__(image_idx) # this loads the whole image every time, could probably be more efficient
        _, img_h, img_w = img.shape

        x1, y1, x2, y2 = index_entry['boxes'][bbox_idx]
        w = min(img_w, x2) - max(0, x1)
        h = min(img_h, y2) - max(0, y1)

        larger_dim = max(w, h)
        res = torch.full((3, larger_dim, larger_dim), 0.5) # 0.5 at 0 - 1 == 0 at 0 - -1
        res[:, 0:h, 0:w] = img[:, y1:y2, x1:x2]

        return ttf.resize(res, (CLASSIFICATION_IMAGE_SIZE, CLASSIFICATION_IMAGE_SIZE))

def gp_collate_fn(samples):
    emb_images, gen_images, categories = zip(*samples) # Returining two sets of images is a bit dirty, TODO check if you have better ideas later
    return torch.stack(emb_images), torch.stack(gen_images), categories

def gp_annotated_collate_fn(samples):
    emb_images, gen_images, categories, annotations = zip(*samples)
    return torch.stack(emb_images), torch.stack(gen_images), categories, annotations

class GroceryProductsDataset(tdata.Dataset): # TODO: Clean this one up a bunch
    def __init__(self, image_roots, skip=[r'^Background.*$', r'^.*/[Oo]riginals?$'], only=None,
        random_crop = True, min_cropped_size = 0.8,
        test_can_load = False, include_annotations = False, include_masks = False, index_from_file = False):
        super().__init__()

        skip = re.compile('|'.join(f'({s})' for s in skip))
        if index_from_file:
            self.paths, self.categories, self.annotations = self.build_index_from_file(image_roots, skip, only)
        else:
            self.paths, self.categories, self.annotations = self.build_index(image_roots, skip, only, test_can_load)
        self.random_crop = random_crop
        self.min_cropped_size = min_cropped_size
        self.include_annotations = include_annotations
        self.include_masks = include_masks
    def build_index(self, image_roots, skip, only, test_can_load):
        print('Building index...')
        paths = []
        categories = []
        annotations = []
        if test_can_load: skipped = []
        for r in image_roots:
            print(f'Processing {r}...')
            to_search = [r]
            hierarchies = [[]]
            while len(to_search):
                current_path = to_search.pop()
                current_hierarchy = hierarchies.pop()
                if skip.match('/'.join(current_hierarchy)) is not None:
                    continue
                if only is not None and len(current_hierarchy) and current_hierarchy[0] not in only:
                    continue
                for entry in os.scandir(current_path):
                    if entry.is_dir(follow_symlinks=False): # not following symlinks here to avoid possibily infinite looping
                        to_search.append(entry.path)
                        hierarchies.append(current_hierarchy + [entry.name])
                    elif entry.is_file():
                        if entry.name in ('.DS_Store', 'index.txt', 'TrainingClassesIndex.mat', 'classes.csv', 'Thumbs.db'): continue
                        if test_can_load:
                            try:
                                pil.Image.open(entry.path)
                            except OSError:
                                skipped.append(entry.path)
                                continue
                        paths.append(entry.path)
                        categories.append(current_hierarchy)
                        annotations.append('/'.join([*current_hierarchy, entry.name]))
            print(f'-> Index size: {len(paths)}')
        print('Index built!')
        if test_can_load and skipped:
            print(f'Skipped a total of {len(skipped)} files due to not being image files openable with pillow')
            if len(skipped) < 10:
                print(f'(Skipped: {skipped})')
        return paths, categories, annotations
    def build_index_from_file(self, dataset_roots, skip, only, index_filename = 'TrainingFiles.txt'):
        print('Building index...')
        paths = []
        categories = []
        annotations = []
        for dataset_root in dataset_roots:
            print(f'Processing {dataset_root}...')
            index_file = path.join(dataset_root, index_filename)
            with open(index_file, 'r') as f:
                for l in f:
                    parts = l.strip().split('/')
                    if len(parts) < 2: continue

                    hier = parts[1:-1] # Skip the first folder, it's always "Training", and the image name
                    if only is not None and hier[0] not in only: continue
                    if skip.match('/'.join(hier)) is not None: continue

                    paths.append(path.join(dataset_root, *parts))
                    categories.append(hier)
                    annotations.append('/'.join(parts[1:]))
            print(f'-> Index size: {len(paths)}')
        print(f'Index built!')
        return paths, categories, annotations
    def tensorize(self, img, tanh=False, mask=False):
        new_size = (CLASSIFICATION_IMAGE_SIZE, round(CLASSIFICATION_IMAGE_SIZE * img.width / img.height)) if img.height > img.width \
            else (round(CLASSIFICATION_IMAGE_SIZE * img.height / img.width), CLASSIFICATION_IMAGE_SIZE)
        img = ttf.resize(img, new_size)

        w, h = img.width, img.height
        img = ttf.to_tensor(img)
        if mask: # TODO: Consider doing this in advance instead of as a part of loading the images
            m = utils.build_mask(img)[None]
            m = ttf.pad(m, (0, 0, CLASSIFICATION_IMAGE_SIZE - w, CLASSIFICATION_IMAGE_SIZE - h), fill=1)
        if tanh:
            img = utils.scale_to_tanh(img)
        img = ttf.pad(img, (0, 0, CLASSIFICATION_IMAGE_SIZE - w, CLASSIFICATION_IMAGE_SIZE - h), fill=0 if tanh else 0.5)
        return torch.cat((img, m)) if mask else img
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        img = pil.Image.open(path)

        if self.random_crop:
            w_ratio = self.min_cropped_size + torch.rand(1) * (1 - self.min_cropped_size)
            min_h_ratio = self.min_cropped_size / w_ratio
            h_ratio = min_h_ratio + torch.rand(1) * (1 - min_h_ratio)
            crop_h = int(img.height * h_ratio)
            crop_w = int(img.width * w_ratio)
            crop_y = torch.randint(0, img.height - crop_h, (1,)).item() if crop_h < img.height else 0
            crop_x = torch.randint(0, img.width - crop_w, (1,)).item() if crop_w < img.width else 0
            gen_img = ttf.crop(img, crop_y, crop_x, crop_h, crop_w)
        else:
            gen_img = img

        if self.include_annotations:
            return self.tensorize(img, True), self.tensorize(gen_img, True, self.include_masks), self.categories[i], self.annotations[i]
        else:
            return self.tensorize(img, True), self.tensorize(gen_img, True, self.include_masks), self.categories[i]

class GroceryProductsTestSet(tdata.Dataset):
    def __init__(self, image_dir, ann_dir, only=None, skip=None):
        if only is not None and skip is not None:
            raise NotImplementedError('Can\'t have both only and skip in GroceryProductsTestSet!')

        super().__init__()
        self.image_dir = image_dir
        self.index = self.build_index(ann_dir, only, skip)
    def get_image_path(self, store, image):
        return path.join(self.image_dir, f'store{store}', 'images', f'store{store}_{image}.jpg')
    def build_index(self, ann_dir, only, skip):
        ann_file_re = re.compile(r'^s(\d+)_(\d+)\.csv$')
        index = []
        for entry in os.scandir(ann_dir):
            if not entry.is_file(): continue

            if only is not None and entry.name not in only: continue
            if skip is not None and entry.name in skip: continue

            match = ann_file_re.match(entry.name)
            if match is None: continue

            anns = []
            boxes = []
            with open(entry.path, 'r') as annotation_file:
                annotation_reader = csv.reader(annotation_file, skipinitialspace=True)
                for row in annotation_reader:
                    if len(row) != 5:
                        print(f'Malformed annotation row in file {entry.name}: {row}; skipping')
                        continue
                    ann, x1, y1, x2, y2 = row
                    anns.append(ann)
                    boxes.append([int(coord) for coord in (x1, y1, x2, y2)])

            index.append({
                'id': (match.group(1), match.group(2)),
                'path': self.get_image_path(match.group(1), match.group(2)),
                'anns': anns,
                'boxes': torch.tensor(boxes),
            })
        
        return index
    def get_index_for(self, store, image):
        target_path = self.get_image_path(store, image)
        for i, idx in enumerate(self.index):
            if idx['path'] == target_path:
                return i
        return None
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        index_entry = self.index[i]
        img = pil.Image.open(index_entry['path'])
        return ttf.to_tensor(img), index_entry['anns'], index_entry['boxes']

class PlanogramTestSet(GroceryProductsTestSet):
    def __init__(self, image_dir, ann_dir, plano_dir, only=None, skip=None):
        self.plano_dir = plano_dir
        super().__init__(image_dir, ann_dir, only, skip)
    def build_index(self, ann_dir, only, skip):
        index = super().build_index(ann_dir, only, skip)
        for entry in index:
            s, i = entry['id']
            plano_path = path.join(self.plano_dir, f's{s}_{i}.json')
            boxes, labels, g = planogram_adapters.read_tonioni_planogram(plano_path)
            entry['plano'] = {
                'boxes': boxes, 'labels': labels, 'graph': g,
            } 
        return index
    def __getitem__(self, i):
        img, anns, boxes = super().__getitem__(i)
        return img, anns, boxes, self.index[i]['plano']

class InternalPlanoSet(tdata.Dataset):
    def __init__(self, dir):
        super().__init__()
        self.index = self.build_index(dir)
    def build_index(self, dir):
        index_path = path.join(dir, 'index.json')
        with open(index_path, 'r') as index_file:
            index = json.load(index_file)

        res = []
        for obj in index:
            img_path = path.join(dir, obj['image'])
            with open(path.join(dir, obj['planogram']), 'r') as plano_file:
                plano = json.load(plano_file)
            anns = [e['code'] for e in plano]
            boxes = torch.tensor([e['box'] for e in plano], dtype=torch.float)
            res.append({
                'img': img_path,
                'anns': anns,
                'boxes': boxes,
            })
        
        return res
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        index_entry = self.index[i]
        img = pil.Image.open(index_entry['img'])
        return ttf.to_tensor(img), {'labels': index_entry['anns'], 'boxes': index_entry['boxes']}
