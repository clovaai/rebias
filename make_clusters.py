"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import argparse
import os
import time

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from datasets.imagenet import get_imagenet_dataloader

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='ImageNet')
parser.add_argument('--num_classes', type=int, default=9, help='number of classes')
parser.add_argument('--load_size', type=int, default=256, help='image load size')
parser.add_argument('--image_size', type=int, default=224, help='image crop size')
parser.add_argument('--k', type=int, default=9, help='number of clusters')
parser.add_argument('--n_sample', type=int, default='30', help='number of samples per cluster')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--cluster_dir', type=str, default='clusters')


def main(n_try=None):
    args = parser.parse_args()

    # create directories if not exist
    if not os.path.exists(args.cluster_dir):
        os.makedirs(args.cluster_dir)

    data_loader = get_imagenet_dataloader(batch_size=args.batch_size, train=False)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    extractor = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features)[:-16])  # conv1_2
    extractor.cuda()

    # ======================================================================= #
    #                         1. Extract features                             #
    # ======================================================================= #
    print('Start extracting features...')
    extractor.eval()
    N = len(data_loader.dataset.dataset)

    start = time.time()
    for i, (images, targets, _) in enumerate(data_loader):
        images = images.cuda()
        outputs = gram_matrix(extractor(images))
        outputs = outputs.view(images.size(0), -1).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, outputs.shape[1])).astype('float32')

        if i < N - 1:
            features[i * args.batch_size: (i+1) * args.batch_size] = outputs.astype('float32')

        else:
            features[i * args.batch_size:] = outputs.astype('float32')

    # L2 normalization
    features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
    print('Finished extracting features...(time: {0:.0f} s)'.format(time.time() - start))

    # ======================================================================= #
    #                             2. Clustering                               #
    # ======================================================================= #
    start = time.time()
    labels, image_lists = Kmeans(args.k, features)
    print('Finished clustering...(time: {0:.0f} s)'.format(time.time() - start))

    # save clustering results
    torch.save(torch.LongTensor(labels), os.path.join(args.cluster_dir,
                                                      'cluster_label_{}.pth'.format(n_try)))
    print('Saved cluster label...')

    len_list = [len(image_list) for image_list in image_lists]
    min_len = min(len_list)
    if min_len < args.n_sample:
        args.n_sample = min_len
    print('number of images in each cluster:', len_list)

    # sample clustering results
    start = time.time()
    samples = [[]] * args.k
    for k in range(args.k):
        idx_list = image_lists[k]  # list of image indexes in each cluster
        for j in range(args.n_sample):  # sample j indexes
            idx = idx_list[j]
            filename = data_loader.dataset.dataset[idx][0]
            image = transform(Image.open(filename).convert('RGB')).unsqueeze(0)
            samples[k] = samples[k] + [image]

    for k in range(args.k):
        samples[k] = torch.cat(samples[k], dim=3)
    samples = torch.cat(samples, dim=0)

    filename = os.path.join(args.cluster_dir, 'cluster_sample_{}.jpg'.format(n_try))
    save_image(denorm(samples.data.cpu()), filename, nrow=1, padding=0)
    print('Finished sampling...(time: {0:.0f} s)'.format(time.time() - start))


def gram_matrix(input, normalize=True):
    N, C, H, W = input.size()
    feat = input.view(N, C, -1)
    G = torch.bmm(feat, feat.transpose(1, 2))  # N X C X C
    if normalize:
        G /= (C * H * W)
    return G


def denorm(x):
    """Convert the range to [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return x.mul_(std[:, None, None]).add_(mean[:, None, None]).clamp_(0, 1)


def Kmeans(k, features):
    n_data, dim = features.shape
    features = torch.FloatTensor(features)

    clus = MiniBatchKMeans(n_clusters=k,
                           batch_size=1024).fit(features)
    labels = clus.labels_

    image_lists = [[] for _ in range(k)]
    feat_lists = [[] for _ in range(k)]
    for i in range(n_data):
        image_lists[labels[i]].append(i)
        feat_lists[labels[i]].append(features[i].unsqueeze(0))

    return labels, image_lists


if __name__ == '__main__':
    for i in range(5):
        main(i+1)
