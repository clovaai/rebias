"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Entry point of Kinetics experiments.
NOTE: We will not handle the issues from action recognition experiments.

This script provides full implementations including
- Various methods (ReBias, Vanilla, Biased, LearnedMixIn, RUBi)
    - Target network: ResNet3D
    - Biased network: ResNet2D
- Sub-sampled 10-Class Kinetics / Mimetics from the full datasets.
    - Please see datasets/kinetics.py for details.

Usage:
    python main_action.py --train_root /path/to/your/kinetics/train
                          --train_annotation_file /path/to/your/kinetics/train_annotion
                          --eval_root /path/to/your/mimetics/train
                          --eval_annotation_file /path/to/your/kinetics/train_annotion

"""
import fire

from datasets.kinetics import get_kinetics_dataloader
from evaluator import ActionEvaluator
from logger import PythonLogger
from trainer import Trainer
from models import ResNet3D, ReBiasModels


class ActionTrainer(Trainer):
    def _set_models(self):
        f_net = ResNet3D.ResNet3DModel(**self.options.f_config)
        g_nets = [ResNet3D.ResNet3DModel(**self.options.g_config)
                  for _ in range(self.options.n_g_nets)]

        self.model = ReBiasModels(f_net, g_nets)
        self.evaluator = ActionEvaluator(device=self.device)


def main(train_root,
         train_annotation_file,
         eval_root,
         eval_annotation_file,
         train_dataset='kinetics10',
         eval_dataset='mimetics10',
         batch_size=128,
         num_classes=10,
         # optimizer config
         lr=0.1,
         optim='Adam',
         n_epochs=120,
         lr_step_size=20,
         scheduler='CosineAnnealingLR',
         n_f_pretrain_epochs=0,
         n_g_pretrain_epochs=0,
         f_lambda_outer=1,
         g_lambda_inner=1,
         n_g_update=1,
         update_g_cls=True,
         # criterion config
         outer_criterion='RbfHSIC',
         inner_criterion='MinusRbfHSIC',
         rbf_sigma_scale_x=2,
         rbf_sigma_scale_y=0.5,
         rbf_sigma_x=1,
         rbf_sigma_y=1,
         update_sigma_per_epoch=False,
         sigma_update_sampling_rate=0.25,
         hsic_alg='unbiased',
         feature_pos='post',
         # model configs
         n_g_nets=1,
         final_bottleneck_dim=0,
         resnet_depth=18,
         f_temporal_kernel_sizes='33333',
         g_temporal_kernel_sizes='11111',
         resnet_base_width=32,
         # logging
         log_step=10,
         ):
    logger = PythonLogger()
    logger.log('preparing val loader...')

    val_loaders = {}
    val_loaders['unbiased'] = get_kinetics_dataloader(root=eval_root, batch_size=batch_size,
                                                      logger=logger,
                                                      anno_file=eval_annotation_file,
                                                      dataset_name=eval_dataset,
                                                      split='test')
    val_loaders['val'] = get_kinetics_dataloader(train_root, batch_size=batch_size,
                                                 logger=logger,
                                                 anno_file=train_annotation_file,
                                                 dataset_name=train_dataset,
                                                 split='val')

    logger.log('preparing train loader...')
    tr_loader = get_kinetics_dataloader(train_root, batch_size=batch_size,
                                        logger=logger,
                                        anno_file=train_annotation_file,
                                        dataset_name=train_dataset,
                                        split='train')

    logger.log('preparing trainer...')

    if scheduler == 'StepLR':
        f_scheduler_config = {'step_size': lr_step_size}
        g_scheduler_config = {'step_size': lr_step_size}
    elif scheduler == 'CosineAnnealingLR':
        f_scheduler_config = {'T_max': n_epochs}
        g_scheduler_config = {'T_max': n_epochs}
    else:
        raise NotImplementedError

    # XXX resnet_base_width should be 32.
    if outer_criterion == 'LearnedMixin':
        outer_criterion_config = {'feat_dim': 256, 'num_classes': num_classes}
    elif outer_criterion == 'RUBi':
        outer_criterion_config = {'feat_dim': 256}
    else:
        outer_criterion_config = {'sigma_x': rbf_sigma_x, 'sigma_y': rbf_sigma_y,
                                  'algorithm': hsic_alg}

    engine = ActionTrainer(
        outer_criterion=outer_criterion,
        inner_criterion=inner_criterion,
        outer_criterion_config=outer_criterion_config,
        outer_criterion_detail={'sigma_x_type': rbf_sigma_x,
                                'sigma_y_type': rbf_sigma_y,
                                'sigma_x_scale': rbf_sigma_scale_x,
                                'sigma_y_scale': rbf_sigma_scale_y},
        inner_criterion_config={'sigma_x': rbf_sigma_x, 'sigma_y': rbf_sigma_y,
                                'algorithm': hsic_alg},
        inner_criterion_detail={'sigma_x_type': rbf_sigma_x,
                                'sigma_y_type': rbf_sigma_y,
                                'sigma_x_scale': rbf_sigma_scale_x,
                                'sigma_y_scale': rbf_sigma_scale_y},
        n_epochs=n_epochs,
        n_f_pretrain_epochs=n_f_pretrain_epochs,
        n_g_pretrain_epochs=n_g_pretrain_epochs,
        f_config={'resnet_depth': resnet_depth,
                  'model_arch': f_temporal_kernel_sizes,
                  'feature_position': feature_pos,
                  'width_per_group': resnet_base_width,
                  'num_classes': num_classes,
                  'final_bottleneck_dim': final_bottleneck_dim
                  },
        g_config={'resnet_depth': resnet_depth,
                  'model_arch': g_temporal_kernel_sizes,
                  'feature_position': feature_pos,
                  'width_per_group': resnet_base_width,
                  'num_classes': num_classes,
                  'final_bottleneck_dim': final_bottleneck_dim
                  },
        optimizer=optim,
        f_optim_config={'lr': lr, 'weight_decay': 1e-4},
        g_optim_config={'lr': lr, 'weight_decay': 1e-4},
        f_scheduler_config=f_scheduler_config,
        g_scheduler_config=g_scheduler_config,
        scheduler=scheduler,
        f_lambda_outer=f_lambda_outer,
        g_lambda_inner=g_lambda_inner,
        n_g_update=n_g_update,
        update_g_cls=update_g_cls,
        n_g_nets=n_g_nets,
        train_loader=tr_loader,
        logger=logger,
        log_step=log_step,
        sigma_update_sampling_rate=sigma_update_sampling_rate)
    engine.train(tr_loader, val_loaders=val_loaders,
                 update_sigma_per_epoch=update_sigma_per_epoch)

    val_loaders['val'] = get_kinetics_dataloader(train_root, batch_size=batch_size,
                                                 logger=logger,
                                                 anno_file=train_annotation_file,
                                                 dataset_name=train_dataset,
                                                 split='test')
    evaluator = ActionEvaluator()
    engine.evaluate(evaluator,
                    step=n_epochs,
                    val_loaders=val_loaders)


if __name__ == '__main__':
    fire.Fire(main)
