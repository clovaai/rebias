"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

9-Class ImageNet wrapper. Many codes are borrowed from the official torchvision dataset.
https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py

The following nine classes are selected to build the subset:
    dog, cat, frog, turtle, bird, monkey, fish, crab, insect
"""
import os
from PIL import Image
from torchvision import transforms
import torch
import torch.utils.data


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

CLASS_TO_INDEX = {'n01641577': 2, 'n01644373': 2, 'n01644900': 2, 'n01664065': 3, 'n01665541': 3,
                  'n01667114': 3, 'n01667778': 3, 'n01669191': 3, 'n01819313': 4, 'n01820546': 4,
                  'n01833805': 4, 'n01843383': 4, 'n01847000': 4, 'n01978287': 7, 'n01978455': 7,
                  'n01980166': 7, 'n01981276': 7, 'n02085620': 0, 'n02099601': 0, 'n02106550': 0,
                  'n02106662': 0, 'n02110958': 0, 'n02123045': 1, 'n02123159': 1, 'n02123394': 1,
                  'n02123597': 1, 'n02124075': 1, 'n02174001': 8, 'n02177972': 8, 'n02190166': 8,
                  'n02206856': 8, 'n02219486': 8, 'n02486410': 5, 'n02487347': 5, 'n02488291': 5,
                  'n02488702': 5, 'n02492035': 5, 'n02607072': 6, 'n02640242': 6, 'n02641379': 6,
                  'n02643566': 6, 'n02655020': 6}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, data='ImageNet'):
    # dog, cat, frog, turtle, bird, monkey, fish, crab, insect
    RESTRICTED_RANGES = [(151, 254), (281, 285), (30, 32), (33, 37), (89, 97),
                         (372, 378), (393, 397), (118, 121), (306, 310)]
    range_sets = [set(range(s, e + 1)) for s, e in RESTRICTED_RANGES]
    class_to_idx_ = {}

    if data == 'ImageNet-A':
        for class_name, idx in class_to_idx.items():
            try:
                class_to_idx_[class_name] = CLASS_TO_INDEX[class_name]
            except Exception:
                pass
    elif data == 'ImageNet-C':
        # TODO
        pass
    else:  # ImageNet
        for class_name, idx in class_to_idx.items():
            for new_idx, range_set in enumerate(range_sets):
                if idx in range_set:
                    if new_idx == 0:  # classes that overlap with ImageNet-A
                        if idx in [151, 207, 234, 235, 254]:
                            class_to_idx_[class_name] = new_idx
                    elif new_idx == 4:
                        if idx in [89, 90, 94, 96, 97]:
                            class_to_idx_[class_name] = new_idx
                    elif new_idx == 5:
                        if idx in [372, 373, 374, 375, 378]:
                            class_to_idx_[class_name] = new_idx
                    else:
                        class_to_idx_[class_name] = new_idx
    images = []
    dir = os.path.expanduser(dir)
    a = sorted(class_to_idx_.keys())
    for target in a:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx_[target])
                    images.append(item)

    return images, class_to_idx_


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader,
                 train=True, val_data='ImageNet'):
        classes, class_to_idx = find_classes(root)
        imgs, class_to_idx_ = make_dataset(root, class_to_idx, val_data)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                   "Supported image extensions are: " + ",".join(
                       IMG_EXTENSIONS)))
        self.root = root
        self.dataset = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx_
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.val_data = val_data
        self.clusters = []
        for i in range(3):
            self.clusters.append(torch.load('clusters/cluster_label_{}.pth'.format(i+1)))

    def __getitem__(self, index):
        path, target = self.dataset[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train and self.val_data == 'ImageNet':
            bias_target = [self.clusters[0][index],
                           self.clusters[1][index],
                           self.clusters[2][index]]
            return img, target, bias_target

        else:
            return img, target, target

    def __len__(self):
        return len(self.dataset)


def get_imagenet_dataloader(root, batch_size, train=True, num_workers=8,
                            load_size=256, image_size=224, val_data='ImageNet'):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    else:
        transform = transforms.Compose([
            transforms.Resize(load_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    dataset = ImageFolder(root, transform=transform, train=train, val_data=val_data)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=train,
                                             num_workers=num_workers,
                                             pin_memory=True)

    return dataloader
