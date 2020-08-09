"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Entry point of 9-Class ImageNet experiments.

This script provides full implementations including
- Various methods (ReBias, Vanilla, Biased, LearnedMixIn, RUBi)
    - Target network: ResNet-18
    - Biased network: BagNet-18
    - We do not provide Stylised ImageNet implementation here. See README.md for details.
- Sub-sampled 9-Class ImageNet / ImageNet-A from the full ImageNet / ImageNet-A folder.
    - Please see datasets/imagenet.py for details.
- Cluster-based unbiased accuracies.
    - For curious readers, `make_clusters.py` shows how to make texture clusters.

Usage:
    python main_imagenet.py --train_root /path/to/your/imagenet/train
                            --val_root /path/to/your/imagenet/val
                            --imageneta_root /path/to/your/imagenet_a
"""
import fire

from datasets.imagenet import get_imagenet_dataloader
from evaluator import ImageNetEvaluator
from logger import PythonLogger
from trainer import Trainer
from models import resnet18, bagnet18, ReBiasModels


class ImageNetTrainer(Trainer):
    def _set_models(self):
        f_net = resnet18(**self.options.f_config)
        g_nets = [bagnet18(**self.options.g_config)
                  for _ in range(self.options.n_g_nets)]

        self.model = ReBiasModels(f_net, g_nets)
        self.evaluator = ImageNetEvaluator(device=self.device)


def main(train_root,
         val_root,
         imageneta_root,
         batch_size=128,
         num_classes=9,
         # optimizer config
         lr=0.001,
         optim='Adam',
         n_epochs=120,
         lr_step_size=30,
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
         rbf_sigma_scale_x=1,
         rbf_sigma_scale_y=1,
         rbf_sigma_x='median',
         rbf_sigma_y='median',
         update_sigma_per_epoch=True,
         hsic_alg='unbiased',
         feature_pos='post',
         # model configs
         n_g_nets=1,
         final_bottleneck_dim=0,
         # logging
         log_step=10,
         # others
         save_dir='./checkpoints',
         ):
    logger = PythonLogger()
    logger.log('preparing train loader...')
    tr_loader = get_imagenet_dataloader(train_root,
                                        batch_size=batch_size,
                                        train=True)

    logger.log('preparing val loader...')
    val_loaders = {}
    val_loaders['biased'] = get_imagenet_dataloader(val_root,
                                                    batch_size=batch_size,
                                                    train=False)
    val_loaders['unbiased'] = get_imagenet_dataloader(val_root,
                                                      batch_size=batch_size,
                                                      train=False)
    val_loaders['imagenet-a'] = get_imagenet_dataloader(imageneta_root,
                                                        batch_size=batch_size,
                                                        train=False,
                                                        val_data='ImageNet-A')

    logger.log('preparing trainer...')

    if scheduler == 'StepLR':
        f_scheduler_config = {'step_size': lr_step_size}
        g_scheduler_config = {'step_size': lr_step_size}
    elif scheduler == 'CosineAnnealingLR':
        f_scheduler_config = {'T_max': n_epochs}
        g_scheduler_config = {'T_max': n_epochs}
    else:
        raise NotImplementedError

    if outer_criterion == 'LearnedMixin':
        outer_criterion_config = {'feat_dim': 512, 'num_classes': 9}
    elif outer_criterion == 'RUBi':
        outer_criterion_config = {'feat_dim': 512}
    else:
        outer_criterion_config = {'sigma_x': rbf_sigma_x, 'sigma_y': rbf_sigma_y,
                                  'algorithm': hsic_alg}

    engine = ImageNetTrainer(
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
        f_config={'feature_pos': feature_pos,
                  'num_classes': num_classes},
        g_config={'feature_pos': feature_pos,
                  'num_classes': num_classes},
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
        log_step=log_step)

    engine.train(tr_loader, val_loaders=val_loaders,
                 val_epoch_step=1,
                 update_sigma_per_epoch=update_sigma_per_epoch,
                 save_dir=save_dir)


if __name__ == '__main__':
    fire.Fire(main)
