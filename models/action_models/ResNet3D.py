import torch.nn as nn
from .weight_init_helper import init_weights
from .stem_helper import VideoModelStem
from .resnet_helper import ResStage
from .head_helper import ResNetBasicHead

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18.1: (2, 2, 2, 2),
                      18: (2, 2, 2, 2),
                      34.1: (3, 4, 6, 3),
                      50: (3, 4, 6, 3),
                      101: (3, 4, 23, 3)}
_MODEL_TRANS_FUNC = {18.1: 'basic_transform',
                     18: 'basic_transform',
                     34.1: 'basic_transform',
                     50: 'bottleneck_transform',
                     101: 'bottleneck_transform'}

# width_multiplier = {18: [1, 1, 2, 4, 8],
#                     50: [1, 4, 8, 16, 32]}
width_multiplier = {18.1: [1, 1, 2, 4, 8],
                    34.1: [1, 1, 2, 4, 8],
                    18: [1, 4, 8, 16, 32],
                    50: [1, 4, 8, 16, 32]}

_POOL1 = [[1, 1, 1]]

_TEMPORAL_KERNEL_BASIS = {
    "11111": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "33333": [
        [[3]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "11133": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
}
FC_INIT_STD = 0.01
ZERO_INIT_FINAL_BN = False
NUM_BLOCK_TEMP_KERNEL = [[2], [2], [2], [2]]

DATA_NUM_FRAMES = 8
DATA_CROP_SIZE = 224

NONLOCAL_LOCATION = [[[]], [[]], [[]], [[]]]
NONLOCAL_GROUP = [[1], [1], [1], [1]]
NONLOCAL_INSTANTIATION = "dot_product"

RESNET_STRIDE_1X1 = False
RESNET_INPLACE_RELU = True


class ResNet3DModel(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, SlowOnly).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self,
                 model_arch='33333',
                 resnet_depth=18,
                 feature_position='post',
                 width_per_group=32,
                 dropout_rate=0.0,
                 num_classes=400,
                 final_bottleneck_dim=0
                 ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet3DModel, self).__init__()
        self.num_pathways = 1
        self._construct_network(
            model_arch=model_arch,
            resnet_depth=resnet_depth,
            dropout_rate=dropout_rate,
            width_per_group=width_per_group,
            num_classes=num_classes,
            feature_position=feature_position,
            final_bottleneck_dim=final_bottleneck_dim
        )
        init_weights(
            self, FC_INIT_STD, ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, model_arch='33333',
                           resnet_depth=18,
                           feature_position='post',
                           num_groups=1,
                           width_per_group=32,
                           input_channel_num=None,
                           dropout_rate=0.0,
                           num_classes=400,
                           final_bottleneck_dim=0):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        if input_channel_num is None:
            input_channel_num = [3]
        pool_size = _POOL1
        assert len({len(pool_size), self.num_pathways}) == 1
        assert resnet_depth in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[resnet_depth]
        trans_func = _MODEL_TRANS_FUNC[resnet_depth]

        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[str(model_arch)]

        self.s1 = VideoModelStem(
            dim_in=input_channel_num,
            dim_out=[width_per_group * width_multiplier[resnet_depth][0]],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
        )

        self.s2 = ResStage(
            dim_in=[width_per_group * width_multiplier[resnet_depth][0]],
            dim_out=[width_per_group * width_multiplier[resnet_depth][1]],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=[1],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=NONLOCAL_LOCATION[0],
            nonlocal_group=NONLOCAL_GROUP[0],
            instantiation=NONLOCAL_INSTANTIATION,
            trans_func_name=trans_func,
            stride_1x1=RESNET_STRIDE_1X1,
            inplace_relu=RESNET_INPLACE_RELU,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = ResStage(
            dim_in=[width_per_group * width_multiplier[resnet_depth][1]],
            dim_out=[width_per_group * width_multiplier[resnet_depth][2]],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=[2],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=NONLOCAL_LOCATION[1],
            nonlocal_group=NONLOCAL_GROUP[1],
            instantiation=NONLOCAL_INSTANTIATION,
            trans_func_name=trans_func,
            stride_1x1=RESNET_STRIDE_1X1,
            inplace_relu=RESNET_INPLACE_RELU,
        )

        self.s4 = ResStage(
            dim_in=[width_per_group * width_multiplier[resnet_depth][2]],
            dim_out=[width_per_group * width_multiplier[resnet_depth][3]],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=NONLOCAL_LOCATION[2],
            nonlocal_group=NONLOCAL_GROUP[2],
            instantiation=NONLOCAL_INSTANTIATION,
            trans_func_name=trans_func,
            stride_1x1=RESNET_STRIDE_1X1,
            inplace_relu=RESNET_INPLACE_RELU,
        )

        self.s5 = ResStage(
            dim_in=[width_per_group * width_multiplier[resnet_depth][3]],
            dim_out=[width_per_group * width_multiplier[resnet_depth][4]],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=[2],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=NONLOCAL_LOCATION[3],
            nonlocal_group=NONLOCAL_GROUP[3],
            instantiation=NONLOCAL_INSTANTIATION,
            trans_func_name=trans_func,
            stride_1x1=RESNET_STRIDE_1X1,
            inplace_relu=RESNET_INPLACE_RELU,
        )

        self.head = ResNetBasicHead(
            dim_in=[width_per_group * width_multiplier[resnet_depth][4]],
            num_classes=num_classes,
            pool_size=[
                [
                    DATA_NUM_FRAMES // pool_size[0][0],
                    DATA_CROP_SIZE // 32 // pool_size[0][1],
                    DATA_CROP_SIZE // 32 // pool_size[0][2],
                ]
            ],
            dropout_rate=dropout_rate,
            feature_position=feature_position,
            final_bottleneck_dim=final_bottleneck_dim
        )

    def forward(self, x, logits_only=False):
        x = [x]
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x, h = self.head(x)

        if logits_only:
            return x
        else:
            return x, h
