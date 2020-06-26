"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Entry point of Biased-MNIST experiments.

This script provides full implementations including
- Various methods (ReBias, Vanilla, Biased, LearnedMixIn, RUBi)
    - Target network: Stacked convolutional networks (kernel_size=7)
    - Biased network: Stacked convolutional networks (kernel_size=1)
    - We do not provide HEX implementation here. See README.md for details.
- Controllable Biased-MNIST experiments by --train_correlation option.
    - Please see datasets/colour_mnist.py for details.

Usage:
    python main_biased_mnist.py --root /path/to/your/dataset --train_correlation 0.999
"""
import fire

from datasets.colour_mnist import get_biased_mnist_dataloader
from evaluator import MNISTEvaluator
from logger import PythonLogger
from trainer import Trainer
from models import SimpleConvNet, ReBiasModels


class MNISTTrainer(Trainer):
    def _set_models(self):
        if not self.options.f_config:
            self.options.f_config = {'kernel_size': 7, 'feature_pos': 'post'}
            self.options.g_config = {'kernel_size': 1, 'feature_pos': 'post'}

        f_net = SimpleConvNet(**self.options.f_config)
        g_nets = [SimpleConvNet(**self.options.g_config) for _ in range(self.options.n_g_nets)]

        self.model = ReBiasModels(f_net, g_nets)
        self.evaluator = MNISTEvaluator(device=self.device)


def main(root,
         batch_size=256,
         train_correlation=0.999,
         n_confusing_labels=9,
         # optimizer config
         lr=0.001,
         optim='Adam',
         n_epochs=80,
         lr_step_size=20,
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
         rbf_sigma_x=1,
         rbf_sigma_y=1,
         update_sigma_per_epoch=False,
         hsic_alg='unbiased',
         feature_pos='post',
         # model configs
         n_g_nets=1,
         f_kernel_size=7,
         g_kernel_size=1,
         # others
         save_dir='./checkpoints',
         ):
    logger = PythonLogger()
    logger.log('preparing train loader...')
    tr_loader = get_biased_mnist_dataloader(root, batch_size=batch_size,
                                            data_label_correlation=train_correlation,
                                            n_confusing_labels=n_confusing_labels,
                                            train=True)
    logger.log('preparing val loader...')
    val_loaders = {}
    val_loaders['biased'] = get_biased_mnist_dataloader(root, batch_size=batch_size,
                                                        data_label_correlation=1,
                                                        n_confusing_labels=n_confusing_labels,
                                                        train=False)
    val_loaders['rho0'] = get_biased_mnist_dataloader(root, batch_size=batch_size,
                                                      data_label_correlation=0,
                                                      n_confusing_labels=9,
                                                      train=False)
    val_loaders['unbiased'] = get_biased_mnist_dataloader(root, batch_size=batch_size,
                                                          data_label_correlation=0.1,
                                                          n_confusing_labels=9,
                                                          train=False)

    logger.log('preparing trainer...')

    log_step = int(100 * 256 / batch_size)

    engine = MNISTTrainer(
        outer_criterion=outer_criterion,
        inner_criterion=inner_criterion,
        outer_criterion_config={'sigma_x': rbf_sigma_x, 'sigma_y': rbf_sigma_y,
                                'algorithm': hsic_alg},
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
        f_config={'num_classes': 10, 'kernel_size': f_kernel_size, 'feature_pos': feature_pos},
        g_config={'num_classes': 10, 'kernel_size': g_kernel_size, 'feature_pos': feature_pos},
        f_lambda_outer=f_lambda_outer,
        g_lambda_inner=g_lambda_inner,
        n_g_update=n_g_update,
        update_g_cls=update_g_cls,
        n_g_nets=n_g_nets,
        optimizer=optim,
        f_optim_config={'lr': lr, 'weight_decay': 1e-4},
        g_optim_config={'lr': lr, 'weight_decay': 1e-4},
        scheduler='StepLR',
        f_scheduler_config={'step_size': lr_step_size},
        g_scheduler_config={'step_size': lr_step_size},
        train_loader=tr_loader,
        log_step=log_step,
        logger=logger)
    engine.train(tr_loader, val_loaders=val_loaders,
                 val_epoch_step=1,
                 update_sigma_per_epoch=update_sigma_per_epoch,
                 save_dir=save_dir)


if __name__ == '__main__':
    fire.Fire(main)
