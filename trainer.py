"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Unified implementation of the de-biasing minimax optimisation by various methods including,
- ReBias (ours, outer_criterion='RbfHSIC', inner criterion='MinusRbfHSIC')
- Vanilla and Biased baselines (f_lambda_outer=0, g_lambda_inner=0)
- Learned Mixin (outer_criterion='LearnedMixin', g_lambda_inner=0, n_g_update=0)
- RUBi (outer_criterion='RUBi', g_lambda_inner=0)

Also, this implementation allows various configurations such as:
- adaptive radius for RBF kernels (see `_set_adaptive_sigma`)
- warm-up before jointly optimisation (n_g_pretrain_epochs, n_f_pretrain_epochs)
- feature position to compute losses (feature_pos in f_config and g_config)
- various biased network configurations (n_g_nets, n_g_update, update_g_cls)

To see the configurations for each experiment, please refer to the following files:
- README.md
- main_biased_mnist.py
- main_imagenet.py
- main_action.py
"""
import itertools
import os

import munch

import torch
import torch.nn as nn

from criterions import get_criterion
from criterions.sigma_utils import median_distance, feature_dimension
from logger import PythonLogger
from optims import get_optim, get_scheduler


def flatten(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)


def cur_step(cur_epoch, idx, N, fmt=None):
    _cur_step = cur_epoch + idx / N
    if fmt:
        return fmt.format(_cur_step)
    else:
        return _cur_step


class Trainer(object):
    """Base wrapper for the de-biasing minimax optimisation to solve.
    ..math:: min_g max_f L_f + lambda_1 ( L_debias (f, g) - L_g)

    In practice, we optimise the following two minimisation problems sequentially:
    .. math::
        min L_f + f_lambda_outer * outer_criterion (f, g)
        min L_g + g_lambda_inner * inner_criterion (f, g)

    Thus, setting f_lambda_outer or g_lambda_inner to zero means only updating classification loss for the optimisation.
    In practice, ours set f_lambda_outer = g_lambda_inner = 1, and comparison methods set f_lambda_outer = 1 and g_lambda_inner = 0.
    Furthermore, we directly implement criterion functions for comparison methods into `outer_criterion` which also optimise classification too.
    In this case, we solely optimise the outer_criterion without the cross entropy loss.

    Parameters
    ----------
    outer_criterion, inner_criterion: str
        Configurations for setting different criterions including
        - ReBias (ours): RbfHSIC, MinusRbfHSIC
        - Vanilla and Biased baselines: -, -
        - Learned Mixin: LearnedMixin, -
        - RUBi: RUBi, -
        where `-` denotes to no outer/inner optimisation.
    outer_criterion_config, inner_criterion_config: dict
        Configuration dict to define criterions, `criterion_fn(**config)`.
    outer_criterion_detail, inner_criterion_detail: dict
        Configurations dict for more details of each criterion.
        In practice, it only contains sigma configurations such as sigma_x_type, sigma_x_scale.
        To set ``adaptive radius'' for RBF kernels, use sigma_x_type='median' (see `_set_adaptive_sigma`)
    f_config, g_config: dict
        Configuration dict for declaring network objects.
    f_lambda_outer: float
        Control parameter for HSIC or other debiasing objective functions on the target network.
        In the experiments, it is always set to one, except ``baseline'' (Vanilla, Biased) cases.
    g_lambda_inner: float
        Control parameter for HSIC or other debiasing objective functions on the biased network.
        ReBias always use one, otherwise it is set to zero.
    n_g_update: int
        The number of g updates for single f update. It could be used if g update is much slower than expected.
        In the experiments, it is always one.
    update_g_cls: boolean
        Flag for updating g cross entropy loss. If False, only debiasing objective is optimised for g.
    n_g_nets: int
        The number of biased networks for the optimisation. The debiasing loss is the summation of the loss computed by each g.
    n_g_pretrain_epochs, n_f_pretrain_epochs: int
        The warm-up epochs for more stable training.
        It is not used for ReBias, but other comparison methods when there is no biased network update (LearnedMixin).
    train_loader: pytorch dataloader
        Used for adaptive kernel updates.
    sigma_update_sampling_rate: float
        Sampling rate for computing the adaptive kernel radius.
        In the experiments, we use 25% of training data points to compute adaptive kernel radius.
    """
    def __init__(self,
                 # criterion settings
                 outer_criterion='RbfHSIC',
                 inner_criterion='MinusRbfHSIC',
                 outer_criterion_config={'sigma': 1.0},
                 outer_criterion_detail={},
                 inner_criterion_config={},
                 inner_criterion_detail={},
                 # network settings
                 f_config={},
                 g_config={},
                 # optimiser settings
                 f_lambda_outer=1,
                 g_lambda_inner=1,
                 n_g_update=1,
                 update_g_cls=True,
                 n_g_nets=1,
                 optimizer='Adam',
                 f_optim_config=None,
                 g_optim_config=None,
                 scheduler='StepLR',
                 f_scheduler_config={'step_size': 20},
                 g_scheduler_config={'step_size': 20},
                 n_g_pretrain_epochs=0,
                 n_f_pretrain_epochs=0,
                 n_epochs=80,
                 log_step=100,
                 # adaptive sigma settings
                 train_loader=None,
                 sigma_update_sampling_rate=0.25,
                 # others
                 device='cuda',
                 logger=None):

        self.device = device
        self.sigma_update_sampling_rate = sigma_update_sampling_rate

        if logger is None:
            logger = PythonLogger()
        self.logger = logger
        self.log_step = log_step

        if f_config['num_classes'] != g_config['num_classes']:
            raise ValueError('num_classes for f and g should be same.')

        self.num_classes = f_config['num_classes']
        options = {
            'outer_criterion': outer_criterion,
            'inner_criterion': inner_criterion,
            'outer_criterion_config': outer_criterion_config,
            'outer_criterion_detail': outer_criterion_detail,
            'inner_criterion_config': inner_criterion_config,
            'inner_criterion_detail': inner_criterion_detail,
            'f_config': f_config,
            'g_config': g_config,
            'f_lambda_outer': f_lambda_outer,
            'g_lambda_inner': g_lambda_inner,
            'n_g_update': n_g_update,
            'update_g_cls': update_g_cls,
            'n_g_nets': n_g_nets,
            'optimizer': optimizer,
            'f_optim_config': f_optim_config,
            'g_optim_config': g_optim_config,
            'scheduler': scheduler,
            'f_scheduler_config': f_scheduler_config,
            'g_scheduler_config': g_scheduler_config,
            'n_g_pretrain_epochs': n_g_pretrain_epochs,
            'n_f_pretrain_epochs': n_f_pretrain_epochs,
            'n_epochs': n_epochs,
        }

        self.options = munch.munchify(options)
        self.evaluator = None

        self._set_models()
        self._to_device()
        self._to_parallel()
        self._set_criterion(train_loader)
        self._set_optimizer()

        self.logger.log('Outer criterion: {}'.format(self.outer_criterion.__class__.__name__))
        self.logger.log(self.options)

    def _set_models(self):
        raise NotImplementedError

    def _to_device(self):
        self.model.f_net = self.model.f_net.to(self.device)
        for i, g_net in enumerate(self.model.g_nets):
            self.model.g_nets[i] = g_net.to(self.device)

    def _to_parallel(self):
        self.model.f_net = torch.nn.DataParallel(self.model.f_net)
        for i, g_net in enumerate(self.model.g_nets):
            self.model.g_nets[i] = torch.nn.DataParallel(g_net)

    def _set_adaptive_sigma(self, train_loader):
        if self.options.outer_criterion_detail.get('sigma_x_type') == 'median':
            self.logger.log('computing sigma from data median')
            sigma_x, sigma_y = median_distance(self.model, train_loader, self.sigma_update_sampling_rate, device=self.device)
        elif self.options.outer_criterion_detail.get('sigma_x_type') == 'dimension':
            sigma_x, sigma_y = feature_dimension(self.model, train_loader, device=self.device)
        else:
            return
        sigma_x_scale = self.options.outer_criterion_detail.get('sigma_x_scale', 1)
        sigma_y_scale = self.options.outer_criterion_detail.get('sigma_y_scale', 1)
        self.options.outer_criterion_config['sigma_x'] = sigma_x * sigma_x_scale
        self.options.outer_criterion_config['sigma_y'] = sigma_y * sigma_y_scale

        self.options.inner_criterion_config['sigma_x'] = sigma_x * sigma_x_scale
        self.options.inner_criterion_config['sigma_y'] = sigma_y * sigma_y_scale
        self.logger.log('current sigma: ({}) * {} ({}) * {}'.format(sigma_x,
                                                                    sigma_x_scale,
                                                                    sigma_y,
                                                                    sigma_y_scale,
                                                                    ))

    def _set_criterion(self, train_loader):
        self._set_adaptive_sigma(train_loader)
        self.outer_criterion = get_criterion(self.options.outer_criterion)(**self.options.outer_criterion_config)
        self.inner_criterion = get_criterion(self.options.inner_criterion)(**self.options.inner_criterion_config)
        self.classification_criterion = nn.CrossEntropyLoss()

    def _set_optimizer(self):
        f_net_parameters = self.model.f_net.parameters()

        if 'fc' in self.outer_criterion.__dict__:
            """[NOTE] for comparison methods (LearnedMixin, RUBi)
            """
            f_net_parameters += list(self.outer_criterion.fc.parameters())

        self.f_optimizer = get_optim(f_net_parameters,
                                     self.options.optimizer,
                                     self.options.f_optim_config)
        self.g_optimizer = get_optim(flatten([g_net.parameters()
                                              for g_net in self.model.g_nets]),
                                     self.options.optimizer,
                                     self.options.g_optim_config)

        self.f_lr_scheduler = get_scheduler(self.f_optimizer,
                                            self.options.scheduler,
                                            self.options.f_scheduler_config)
        self.g_lr_scheduler = get_scheduler(self.g_optimizer,
                                            self.options.scheduler,
                                            self.options.g_scheduler_config)

    def pretrain(self, dataloader, val_loaders=None):
        for cur_epoch in range(self.options.n_g_pretrain_epochs):
            if self.options.n_epochs == 0:
                self.g_lr_scheduler.step()
            for idx, (x, labels, _) in enumerate(dataloader):
                x = x.to(self.device)
                labels = labels.to(self.device)

                loss_dict = {'step': cur_step(cur_epoch, idx, len(dataloader))}
                self._update_g(x, labels, update_inner_loop=False,
                               loss_dict=loss_dict, prefix='pretrain__')

                if (idx + 1) % self.log_step == 0:
                    self.logger.report(loss_dict,
                                       prefix='[Pretrain G] Report @step: ')

            self.evaluate_acc(cur_epoch + 1,
                              f_acc=False,
                              val_loaders=val_loaders)

        for cur_epoch in range(self.options.n_f_pretrain_epochs):
            if self.options.n_epochs == 0:
                self.f_lr_scheduler.step()
            for idx, (x, labels, _) in enumerate(dataloader):
                x = x.to(self.device)
                labels = labels.to(self.device)

                loss_dict = {'step': cur_step(cur_epoch, idx, len(dataloader))}
                self._update_f(x, labels, update_outer_loop=False,
                               loss_dict=loss_dict, prefix='pretrain__')

                if (idx + 1) % self.log_step == 0:
                    self.logger.report(loss_dict,
                                       prefix='[Pretrain F] Report @step: ')

            self.evaluate_acc(cur_epoch + 1,
                              f_acc=True,
                              val_loaders=val_loaders)

    def _update_g(self, x, labels, update_inner_loop=True, loss_dict=None, prefix=''):
        if loss_dict is None:
            loss_dict = {}

        self.model.train()

        g_loss = 0
        for g_idx, g_net in enumerate(self.model.g_nets):
            preds, g_feats = g_net(x)

            _g_loss = 0
            if self.options.update_g_cls:
                _g_loss_cls = self.classification_criterion(preds, labels)
                _g_loss += _g_loss_cls

                loss_dict['{}g_{}_cls'.format(prefix, g_idx)] = _g_loss_cls.item()

            if update_inner_loop and self.options.g_lambda_inner:
                _, f_feats = self.model.f_net(x)
                _g_loss_inner = self.inner_criterion(g_feats, f_feats, labels=labels)
                _g_loss += self.options.g_lambda_inner * _g_loss_inner

                loss_dict['{}g_{}_inner'.format(prefix, g_idx)] = _g_loss_inner.item()

            g_loss += _g_loss

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        loss_dict['{}g_loss'.format(prefix)] = g_loss.item()

    def _update_f(self, x, labels, update_outer_loop=True, loss_dict=None, prefix=''):
        if loss_dict is None:
            loss_dict = {}

        self.model.train()

        f_loss = 0
        preds, f_feats = self.model.f_net(x)

        if self.options.outer_criterion not in ('LearnedMixin', 'RUBi'):
            """[NOTE] Comparison methods (LearnedMixin, RUBi) do not compute f_loss_cls
            """
            f_loss_cls = self.classification_criterion(preds, labels)
            f_loss += f_loss_cls
            loss_dict['{}f_loss_cls'.format(prefix)] = f_loss_cls.item()

        if update_outer_loop and self.options.f_lambda_outer:
            f_loss_indep = 0
            for g_idx, g_net in enumerate(self.model.g_nets):
                _g_preds, _g_feats = g_net(x)

                _f_loss_indep = self.outer_criterion(f_feats, _g_feats, labels=labels, f_pred=preds)
                f_loss_indep += _f_loss_indep

                loss_dict['{}f_loss_indep_g_{}'.format(prefix, g_idx)] = _f_loss_indep.item()

            f_loss += self.options.f_lambda_outer * f_loss_indep
            loss_dict['{}f_loss_indep'.format(prefix)] = f_loss_indep.item()

        self.f_optimizer.zero_grad()
        f_loss.backward()
        self.f_optimizer.step()

        loss_dict['{}f_loss'.format(prefix)] = f_loss.item()

    def _train_epoch(self, dataloader, cur_epoch):
        for idx, (x, labels, _) in enumerate(dataloader):
            x = x.to(self.device)
            labels = labels.to(self.device)

            loss_dict = {'step': cur_step(cur_epoch, idx, len(dataloader))}
            for _ in range(self.options.n_g_update):
                self._update_g(x, labels, loss_dict=loss_dict, prefix='train__')
            self._update_f(x, labels, loss_dict=loss_dict, prefix='train__')

            if (idx + 1) % self.log_step == 0:
                self.logger.report(loss_dict,
                                   prefix='[Train] Report @step: ')

    def train(self, tr_loader,
              val_loaders=None,
              val_epoch_step=20,
              update_sigma_per_epoch=False,
              save_dir='./checkpoints',
              experiment=None):
        if val_loaders:
            if not isinstance(val_loaders, dict):
                raise TypeError('val_loaders should be dict, not {}'
                                .format(type(val_loaders)))
            if 'unbiased' not in val_loaders:
                raise ValueError('val_loaders should contain key "unbiased", cur keys({})'
                                 .format(list(val_loaders.keys())))
        os.makedirs(save_dir, exist_ok=True)

        self.logger.log('start pretraining')
        self.pretrain(tr_loader, val_loaders=val_loaders)

        best_acc = 0
        self.logger.log('start training')

        for cur_epoch in range(self.options.n_epochs):
            self._train_epoch(tr_loader, cur_epoch)
            self.f_lr_scheduler.step()
            self.g_lr_scheduler.step()
            self.logger.log('F learning rate: {}, G learning rate: {}'.format(
                self.f_lr_scheduler.get_lr(),
                self.g_lr_scheduler.get_lr()
            ))

            metadata = {
                'cur_epoch': cur_epoch + 1,
                'best_acc': best_acc,
            }

            if val_loaders and (cur_epoch + 1) % val_epoch_step == 0:
                scores = self.evaluate(cur_epoch + 1, val_loaders)
                metadata['scores'] = scores

                if scores['unbiased']['f_acc'] > best_acc:
                    metadata['best_acc'] = scores['unbiased']['f_acc']
                    self.save_models(os.path.join(save_dir, 'best.pth'),
                                     metadata=metadata)
            self.save_models(os.path.join(save_dir, 'last.pth'),
                             metadata=metadata)

            if update_sigma_per_epoch:
                self.logger.log('sigma update')
                self._set_criterion(tr_loader)
                sigma_x = self.options.inner_criterion_config['sigma_x']
                sigma_y = self.options.inner_criterion_config['sigma_y']
                self.logger.report({'step': cur_epoch + 1,
                                    'sigma__f': sigma_x,
                                    'sigma__g': sigma_y}, prefix='[Validation] Report @step: ')

    def evaluate(self, step=0, val_loaders=None):
        if not val_loaders:
            return {}

        scores = {}
        for key, val_loader in val_loaders.items():
            scores[key] = self.evaluator.evaluate_rebias(val_loader, self.model,
                                                         outer_criterion=self.outer_criterion,
                                                         inner_criterion=self.inner_criterion,
                                                         num_classes=self.num_classes,
                                                         key=key)

        for key, score in scores.items():
            msg_dict = {'val__{}_{}'.format(key, k): v for k, v in score.items()}
            msg_dict['step'] = step
            self.logger.report(msg_dict, prefix='[Validation] Report @step: ')

        print(scores)
        return scores

    def evaluate_acc(self, step=0, f_acc=True, val_loaders=None):
        if not val_loaders:
            return {}

        scores = {}
        for key, val_loader in val_loaders.items():
            if f_acc:
                scores[key] = self.evaluator.evaluate_acc(val_loader, self.model.f_net)
            else:
                scores[key] = {}
                for idx, g_net in enumerate(self.model.g_nets):
                    scores[key][idx] = self.evaluator.evaluate_acc(val_loader, g_net)

        for key, score in scores.items():
            if f_acc:
                msg_dict = {'pretrain__{}_f_acc'.format(key): score}
            else:
                msg_dict = {'pretrain__{}_g_{}_acc'.format(key, idx): _score for idx, _score in score.items()}

            msg_dict['step'] = step
            self.logger.report(msg_dict, prefix='[Pretrain Validation] Report @step: ')

        return scores

    def save_models(self, save_to, metadata=None):
        state_dict = {
            'f_net': self.model.f_net.state_dict(),
            'g_nets': [g_net.state_dict() for g_net in self.model.g_nets],
            'f_optimizer': self.f_optimizer.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'f_lr_scheduler': self.f_lr_scheduler.state_dict(),
            'g_lr_scheduler': self.g_lr_scheduler.state_dict(),
            'options': dict(self.options),
            'metadata': metadata,
        }
        torch.save(state_dict, save_to)
        self.logger.log('state dict is saved to {}, metadata: {}'.format(
            save_to, metadata))
