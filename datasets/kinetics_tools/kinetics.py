#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import json
import random
import torch
import torch.utils.data

import datasets.kinetics_tools.decoder as decoder
import datasets.kinetics_tools.video_container as container
import datasets.kinetics_tools.transform as transform

import tqdm

DATA_MEAN = [0.45, 0.45, 0.45]
DATA_STD = [0.225, 0.225, 0.225]
TRAIN_JITTER_SCALES = [256, 320]
TRAIN_CROP_SIZE = 224
TEST_CROP_SIZE = 256
TEST_NUM_ENSEMBLE_VIEWS = 10
TEST_NUM_SPATIAL_CROPS = 1
DATA_SAMPLING_RATE = 8
DATA_NUM_FRAMES = 8


class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, root, mode, logger, num_retries=10,
                 dataset_name="kinetics50",
                 anno_file="kinetics-400.json"):
        """
        Construct the Kinetics video loader
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.root = root
        self.anno_file = anno_file
        self.dataset_name = dataset_name

        assert self.dataset_name in ['kinetics400', 'kinetics50', 'mimetics50',
                                     'kinetics10', 'mimetics10',]

        self.logger = logger

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    TEST_NUM_ENSEMBLE_VIEWS * TEST_NUM_SPATIAL_CROPS
            )

        self.logger.log("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _parse_json(self, json_path, valid=False):
        self.logger.log(json_path)
        with open(json_path, 'r') as data_file:
            self.json_data = json.load(data_file)

        if valid:
            c_tr, c_te, c_v = 0, 0, 0
            for jd in self.json_data['database']:
                if self.json_data['database'][jd]['subset'] == 'training':
                    c_tr += 1
                elif self.json_data['database'][jd]['subset'] == 'testing':
                    c_te += 1
                elif self.json_data['database'][jd]['subset'] == 'validation':
                    c_v += 1
            self.logger.log('Number of Training samples: %d' % c_tr)
            self.logger.log('Number of Validation samples: %d' % c_v)
            self.logger.log('Number of Testing samples: %d' % c_te)

    def _get_class_idx_map(self, classes):
        self.class_labels_map = {}
        for index, class_label in enumerate(classes):
            self.class_labels_map[class_label] = index

    def _get_action_label(self, data):
        if self.mode == 'train' and data['subset'] == 'training':
            action_label = data['annotations']['label']
        elif self.mode == 'val' and data['subset'] == 'validation':
            action_label = data['annotations']['label']
        elif self.mode == 'test' and data['subset'] == 'validation':
            action_label = data['annotations']['label']
        elif self.mode == 'test' and data['subset'] == 'testing':
            action_label = None
        else:
            action_label = None
        return action_label

    def _set_path_prefix(self):
        if self.mode == 'train':
            self.PATH_PREFIX = 'train'
        elif self.mode == 'val':
            self.PATH_PREFIX = 'val'
        elif self.mode == 'test':
            self.PATH_PREFIX = 'val'
        else:
            raise NotImplementedError

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(self.root, self.anno_file)

        self._set_path_prefix()

        if path_to_file.endswith('.json'):
            self._parse_json(json_path=path_to_file)


            self._path_to_videos = []
            self._labels = []
            self._spatial_temporal_idx = []
            clip_idx = 0
            num_missing_videos = 0
            subclasses = []
            if self.dataset_name == 'kinetics50':
                with open('datasets/mimetics/mimetics_v1.0_clsannot.txt') as f_subclasses:
                    f_subclasses.readline()
                    for line in f_subclasses.readlines():
                        subclasses.append(line.split()[0])
                self._get_class_idx_map(subclasses)
            elif self.dataset_name == 'kinetics10':
                with open('datasets/mimetics/mimetics_v1.0_clsannot.txt') as f_subclasses:
                    f_subclasses.readline()
                    line_idx = 0
                    for line in f_subclasses.readlines():
                        line_idx += 1
                        if line_idx % 5 == 0:
                            subclasses.append(line.split()[0])
                    print (subclasses)
                self._get_class_idx_map(subclasses)
            else:
                self._get_class_idx_map(self.json_data['labels'])


            for key in tqdm.tqdm(self.json_data['database']):
                data = self.json_data['database'][key]
                action_label = self._get_action_label(data)
                if (action_label not in subclasses) and len(subclasses):
                    continue

                if action_label is None:
                    # when the json_data['subset'] is not matched with 'self.mode', skip this data.
                    # (for example, self.mode=='train' but data['subset']=='testing')
                    continue

                # path = os.path.join(root_path, self.PATH_PREFIX, action_label, key + '.mp4')
                vid_name = key[:-14]

                # possible path lists (.mp4, .mkv, etc.)
                paths = []
                paths.append(os.path.join(self.root, self.PATH_PREFIX, action_label.replace(' ', '_'), key + '.mp4'))
                paths.append(
                    os.path.join(self.root, self.PATH_PREFIX, action_label.replace(' ', '_'), vid_name + '.mp4'))
                paths.append(os.path.join(self.root, self.PATH_PREFIX, action_label.replace(' ', '_'), key + '.mkv'))
                paths.append(
                    os.path.join(self.root, self.PATH_PREFIX, action_label.replace(' ', '_'), vid_name + '.mkv'))
                paths.append(
                    os.path.join(self.root, self.PATH_PREFIX, action_label.replace(' ', '_'), key + '.mp4.mkv'))
                paths.append(
                    os.path.join(self.root, self.PATH_PREFIX, action_label.replace(' ', '_'), vid_name + '.mp4.mkv'))
                paths.append(
                    os.path.join(self.root, self.PATH_PREFIX, action_label.replace(' ', '_'), key + '.mp4.webm'))
                paths.append(
                    os.path.join(self.root, self.PATH_PREFIX, action_label.replace(' ', '_'), vid_name + '.mp4.webm'))

                exist_path = [p for p in paths if os.path.exists(p)]

                label = self.class_labels_map[action_label]
                if len(exist_path) > 0:
                    path = exist_path[0]
                else:
                    # print(path)
                    num_missing_videos += 1
                    continue

                for idx in range(self._num_clips):
                    self._path_to_videos.append(path)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
                clip_idx += 1

            self.logger.log('num_missing_videos: %d' % num_missing_videos)
            # assert (
            #         len(self._path_to_videos) > 0
            # ), "Failed to load Kinetics split {} from {}".format(
            #     self._split_idx, path_to_file
            # )
            self.logger.log(
                "Constructing kinetics_tools dataloader (size: {}) from {}".format(
                    len(self._path_to_videos), path_to_file
                )
            )

        else:
            # path_to_file = os.path.join(
            #     self.root, "{}.csv".format(self.mode)
            # )
            self._path_to_videos = []
            self._labels = []
            self._spatial_temporal_idx = []

            label_strings = []
            with open(path_to_file, "r") as f:
                f.readline()
                for clip_idx, path_label in enumerate(f.read().splitlines()):
                    label_strings.append(path_label.split(',')[0])

            label_strings = sorted(set(label_strings))

            if self.dataset_name == 'mimetics10':
                label_strings = label_strings[4::5]
                print (label_strings)

            with open(path_to_file, "r") as f:
                f.readline()
                for clip_idx, path_label in enumerate(f.read().splitlines()):
                    # assert len(path_label.split()) == 2
                    label_str, path, start_time, end_time, _, _ = path_label.split(',')

                    if self.dataset_name == 'mimetics10' and label_str not in label_strings:
                        continue

                    label = label_strings.index(label_str)
                    path = os.path.join(self.root,
                                        'data',
                                        label_str,
                                        '{0}_{1:06d}_{2:06d}.mp4'.format(path, int(start_time), int(end_time)))
                    if not os.path.exists(path):
                        self.logger.log('{} is not exists!'.format(path))
                        continue

                    for idx in range(self._num_clips):
                        self._path_to_videos.append(path)
                        self._labels.append(int(label))
                        self._spatial_temporal_idx.append(idx)
                        self._video_meta[clip_idx * self._num_clips + idx] = {}
            # assert (
            #         len(self._path_to_videos) > 0
            # ), "Failed to load Kinetics split {} from {}".format(
            #     self._split_idx, path_to_file
            # )
            self.logger.log(
                "Constructing kinetics_tools dataloader (size: {}) from {}".format(
                    len(self._path_to_videos), path_to_file
                )
            )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = TRAIN_JITTER_SCALES[0]
            max_scale = TRAIN_JITTER_SCALES[1]
            crop_size = TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                    self._spatial_temporal_idx[index]
                    // TEST_NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % TEST_NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index]
                )
            except Exception as e:
                self.logger.log(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                DATA_SAMPLING_RATE,
                DATA_NUM_FRAMES,
                temporal_sample_index,
                TEST_NUM_ENSEMBLE_VIEWS,
                # video_meta=self._video_meta[index],
                target_fps=30,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                self.logger.log(self._path_to_videos[index])
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            frames = frames - torch.tensor(DATA_MEAN)
            frames = frames / torch.tensor(DATA_STD)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
            )

            # frames = [frames]
            label = self._labels[index]
            return frames, label, index
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = transform.random_crop(frames, crop_size)
            frames = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
