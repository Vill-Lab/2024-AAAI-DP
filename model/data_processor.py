import os, glob
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms as T
from copy import deepcopy

class InferenceDataset(object):
    def __init__(self, test_description, test_reference_attribute_folder, tokenizer, attribute_transforms, img_token="<|image|>", MAX_REFERENCE_ATTRIBUTE_NUM=3, device=None):
        self.test_description = test_description
        self.test_reference_attribute_folder = test_reference_attribute_folder
        self.tokenizer = tokenizer
        self.img_token = img_token
        self.attribute_transforms = attribute_transforms

        tokenizer.add_tokens([img_token], special_tokens=True)
        self.img_token_id = tokenizer.convert_tokens_to_ids(img_token)
        self.MAX_REFERENCE_ATTRIBUTE_NUM = MAX_REFERENCE_ATTRIBUTE_NUM
        self.device = device
        self.img_ids = None

    def get_description(self, description):
        self.test_description = description

    def get_reference_attribute_folder(self, reference_attribute_folder):
        self.test_reference_attribute_folder = reference_attribute_folder

    def get_img_ids(self, img_ids=None):
        self.img_ids = img_ids

    def get_data(self):
        return self.process_data()

    def process_noun_phrases_ends(self, description):
        input_ids = self.tokenizer.encode(description)

        npem = [False for _ in input_ids]
        tmp_input_ids = []
        tmp_idx = 0

        for i, id in enumerate(input_ids):
            if id == self.img_token_id:
                npem[tmp_idx - 1] = True
            else:
                tmp_input_ids.append(id)
                tmp_idx += 1

        max_len = self.tokenizer.MODEL_MAX_LENGTH

        if len(tmp_input_ids) > max_len:
            tmp_input_ids = tmp_input_ids[:max_len]
        else:
            tmp_input_ids = tmp_input_ids + [self.tokenizer.pad_token_id] * (max_len - len(tmp_input_ids))

        if len(npem) > max_len:
            npem = npem[:max_len]
        else:
            npem = npem + [False] * (max_len - len(npem))

        tmp_input_ids = torch.tensor(tmp_input_ids, dtype=torch.long)
        npem = torch.tensor(npem, dtype=torch.bool)
        return tmp_input_ids.unsqueeze(0), npem.unsqueeze(0)

    def process_data(self):
        attribute_v = []
        img_ids = []

        for img_id in self.img_ids:
            reference_attribute_img_path = sorted(glob.glob(os.path.join(self.test_reference_attribute_folder, img_id, "*.jpg")) + glob.glob(os.path.join(self.test_reference_attribute_folder, img_id, "*.png")) + glob.glob(os.path.join(self.test_reference_attribute_folder, img_id, "*.jpeg")))[0]

            reference_attribute_img = self.attribute_transforms(read_image(reference_attribute_img_path, mode=ImageReadMode.RGB)).to(self.device)
            attribute_v.append(reference_attribute_img)
            img_ids.append(img_id)

        input_ids, img_token_mask = self.process_noun_phrases_ends(self.test_description)

        img_token_idx, img_token_idx_mask = get_img_token_idx(img_token_mask, self.MAX_REFERENCE_ATTRIBUTE_NUM)

        reference_attribute_num = img_token_idx_mask.sum().item()

        attribute_v = torch.stack(attribute_v)  # [MAX_REFERENCE_ATTRIBUTE_NUM, 3, 256, 256]
        attribute_v = attribute_v.to(mem_format=torch.contiguous_format).float()

        return {
            "input_ids": input_ids,
            "img_token_mask": img_token_mask,
            "img_token_idx": img_token_idx,
            "img_token_idx_mask": img_token_idx_mask,
            "attribute_v": attribute_v,
            "reference_attribute_num": torch.tensor(reference_attribute_num),
            "filenames": img_ids,
        }


class DiversePersonDataset(torch.utils.data.Dataset):
    def __init__(self, root, tokenizer, train_transforms, attribute_transforms, attribute_processor, device=None, MAX_REFERENCE_ATTRIBUTE_NUM=3, img_token_num=1, img_token="", reference_attribute_types=None, split="all"):
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = train_transforms
        self.attribute_transforms = attribute_transforms
        self.attribute_processor = attribute_processor
        self.MAX_REFERENCE_ATTRIBUTE_NUM = MAX_REFERENCE_ATTRIBUTE_NUM
        self.img_token = img_token
        self.img_token_num = img_token_num
        self.device = device
        self.reference_attribute_types = reference_attribute_types

        if split == "all":
            img_ids_path = os.path.join(root, "img_ids.txt")
        elif split == "train":
            img_ids_path = os.path.join(root, "img_ids_train.txt")
        elif split == "test":
            img_ids_path = os.path.join(root, "img_ids_test.txt")
        else:
            raise ValueError(f"Unsupported split {split}")

        with open(img_ids_path, "r") as f:
            self.img_ids = f.read().splitlines()

        tokenizer.add_tokens([img_token], special_tokens=True)
        self.img_token_id = tokenizer.convert_tokens_to_ids(img_token)

    def __len__(self):
        return len(self.img_ids)

    

class Processor4SegmentBase(torch.nn.Module):
    def forward(self, img, background, seg_map, id, bbox):
        mask = seg_map != id
        img[:, mask] = background[:, mask]
        h1, w1, h2, w2 = bbox
        return img[:, w1:w2, h1:h2]

    def get_background(self, img):
        raise NotImplementedError


class Processor4SegmentRandom(Processor4SegmentBase):
    def get_background(self, img):
        background = torch.randint(0, 255, img.shape, dtype=img.dtype, device=img.device)
        return background


class Processor4CropTop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor):
        _, h, w = img.shape
        if w >= h:
            return img
        return img[:, :w, :]


class Processor4ZoomInRandom(torch.nn.Module):
    def __init__(self, zoomin_min=1.0, zoomin_max=1.5):
        super().__init__()
        self.zoomin_min = zoomin_min
        self.zoomin_max = zoomin_max

    def forward(self, img: torch.Tensor):
        zoomin = torch.rand(1) * (self.zoomin_max - self.zoomin_min) + self.zoomin_min
        img = T.functional.resize(img, (int(zoomin * img.shape[1]), int(zoomin * img.shape[2])), interpolation=T.InterpolationMode.BILINEAR, antialias=True,)
        img = Processor4CropTop()(img)
        return img


class Processor4CenterCrop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor):
        _, h, w = img.shape
        if h > w:
            pad = (h - w) // 2
            img = torch.nn.functional.pad(img, (pad, pad, 0, 0), "constant", 0)
            img = T.functional.resize(img, (w, w), interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        else:
            pad = (w - h) // 2
            img = img[:, :, pad : pad + h]
        return img


class Transform4TrainWithSegmap(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.img_resize = T.Resize(args.resolution, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        self.seg_map_resize = T.Resize(args.resolution, interpolation=T.InterpolationMode.NEAREST)
        self.flip = T.RandomHorizontalFlip()
        self.crop = Processor4CenterCrop()

    def forward(self, img, seg_map):
        img = self.img_resize(img)
        seg_map = seg_map.unsqueeze(0)
        seg_map = self.seg_map_resize(seg_map)
        img_all = torch.cat([img, seg_map], dim=0)
        img_all = self.flip(img_all)
        img_all = self.crop(img_all)
        img = img_all[:3]
        seg_map = img_all[3:]
        img = (img.float() / 127.5) - 1
        seg_map = seg_map.squeeze(0)
        return img, seg_map


class Transform4TestWithSegmap(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.img_resize = T.Resize(args.resolution, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        self.seg_map_resize = T.Resize(args.resolution, interpolation=T.InterpolationMode.NEAREST)
        self.crop = Processor4CenterCrop()

    def forward(self, img, seg_map):
        img = self.img_resize(img)
        seg_map = seg_map.unsqueeze(0)
        seg_map = self.seg_map_resize(seg_map)
        img_all = torch.cat([img, seg_map], dim=0)
        img_all = self.crop(img_all)
        img = img_all[:3]
        seg_map = img_all[3:]
        img = (img.float() / 127.5) - 1
        seg_map = seg_map.squeeze(0)
        return img, seg_map

def get_img_token_idx(img_token_mask, MAX_REFERENCE_ATTRIBUTE_NUM):
    img_token_idx = torch.nonzero(img_token_mask, as_tuple=True)[1]
    img_token_idx_mask = torch.ones_like(img_token_idx, dtype=torch.bool)
    if len(img_token_idx) < MAX_REFERENCE_ATTRIBUTE_NUM:
        img_token_idx = torch.cat([img_token_idx, torch.zeros(MAX_REFERENCE_ATTRIBUTE_NUM - len(img_token_idx), dtype=torch.long)])
        img_token_idx_mask = torch.cat([img_token_idx_mask,torch.zeros(MAX_REFERENCE_ATTRIBUTE_NUM - len(img_token_idx_mask), dtype=torch.bool)])

    img_token_idx = img_token_idx.unsqueeze(0)
    img_token_idx_mask = img_token_idx_mask.unsqueeze(0)
    return img_token_idx, img_token_idx_mask
