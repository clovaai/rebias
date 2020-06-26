"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import numpy as np


def n_correct(pred, labels):
    _, predicted = torch.max(pred.data, 1)
    n_correct = (predicted == labels).sum().item()
    return n_correct


class EvaluatorBase(object):
    def __init__(self, device='cuda'):
        self.device = device

    @torch.no_grad()
    def evaluate_acc(self, dataloader, model):
        model.eval()

        total = 0
        correct = 0

        for x, labels, index in dataloader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            pred = model(x, logits_only=True)

            batch_size = labels.size(0)
            total += batch_size
            correct += n_correct(pred, labels)

        return correct / total

    @torch.no_grad()
    def evaluate_rebias(self, dataloader, rebias_model,
                        outer_criterion=None,
                        inner_criterion=None,
                        **kwargs):
        raise NotImplementedError


class MNISTEvaluator(EvaluatorBase):
    def _confusion_matrix(self, pred, bias_labels, labels, n_correct, n_total):
        for bias_label in range(10):
            for label in range(10):
                b_indices = (bias_labels.squeeze() == bias_label).nonzero().squeeze()
                t_indices = (labels.squeeze() == label).nonzero().squeeze()

                indices = np.intersect1d(b_indices.detach().cpu().numpy(),
                                         t_indices.detach().cpu().numpy())
                indices = torch.cuda.LongTensor(indices)
                if indices.nelement() == 0:
                    continue
                _n = len(indices)
                _output = pred.index_select(dim=0, index=indices)
                _, predicted = torch.max(_output.data, 1)
                _n_correct = (predicted == labels[indices]).sum().item()

                n_correct[label][bias_label] += _n_correct
                n_total[label][bias_label] += _n
        return n_correct, n_total

    def get_confusion_matrix(self, dataloader, rebias_model):
        n_correct_arr = np.zeros((10, 10))
        n_total = np.zeros((10, 10))

        total = 0
        f_correct = 0
        for x, labels, bias_labels in dataloader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            bias_labels = bias_labels.to(self.device)

            f_pred, g_preds, f_feat, g_feats = rebias_model(x)
            n_correct_arr, n_total = self._confusion_matrix(f_pred, bias_labels, labels, n_correct_arr, n_total)

            f_correct += n_correct(f_pred, labels)
            total += len(labels)
        print('accuracy:', f_correct / total)
        CM = n_correct_arr / (n_total + 1e-12)
        return CM

    @torch.no_grad()
    def evaluate_rebias(self, dataloader, rebias_model,
                        outer_criterion=None,
                        inner_criterion=None,
                        **kwargs):
        rebias_model.eval()

        total = 0
        f_correct = 0
        g_corrects = [0 for _ in rebias_model.g_nets]

        if outer_criterion.__class__.__name__ in ['LearnedMixin', 'RUBi']:
            """For computing HSIC loss only.
            """
            outer_criterion = None

        outer_loss = [0 for _ in rebias_model.g_nets]
        inner_loss = [0 for _ in rebias_model.g_nets]

        for x, labels, _ in dataloader:
            x = x.to(self.device)
            labels = labels.to(self.device)

            f_pred, g_preds, f_feat, g_feats = rebias_model(x)

            batch_size = labels.size(0)
            total += batch_size

            f_correct += n_correct(f_pred, labels)
            for idx, g_pred in enumerate(g_preds):
                g_corrects[idx] += n_correct(g_pred, labels)

            if outer_criterion:
                for idx, g_pred in enumerate(g_preds):
                    outer_loss[idx] += batch_size * outer_criterion(f_pred, g_pred).item()

            if inner_criterion:
                for idx, g_pred in enumerate(g_preds):
                    inner_loss[idx] += batch_size * inner_criterion(f_pred, g_pred).item()

        ret = {'f_acc': f_correct / total}
        for idx, (_g_correct, _outer_loss, _inner_loss) in enumerate(zip(g_corrects, outer_loss, inner_loss)):
            ret['g_{}_acc'.format(idx)] = _g_correct / total
            ret['outer_{}_loss'.format(idx)] = _outer_loss / total
            ret['inner_{}_loss'.format(idx)] = _inner_loss / total
        return ret


class ImageNetEvaluator(EvaluatorBase):
    def imagenet_unbiased_accuracy(self, outputs, labels, cluster_labels,
                                   num_correct, num_instance,
                                   num_cluster_repeat=3):
        for j in range(num_cluster_repeat):
            for i in range(outputs.size(0)):
                output = outputs[i]
                label = labels[i]
                cluster_label = cluster_labels[j][i]

                _, pred = output.topk(1, 0, largest=True, sorted=True)
                correct = pred.eq(label).view(-1).float()

                num_correct[j][label][cluster_label] += correct.item()
                num_instance[j][label][cluster_label] += 1

        return num_correct, num_instance

    @torch.no_grad()
    def evaluate_rebias(self, dataloader, rebias_model,
                        outer_criterion=None,
                        inner_criterion=None,
                        num_classes=9,
                        num_clusters=9,
                        num_cluster_repeat=3,
                        key=None):
        rebias_model.eval()

        total = 0
        f_correct = 0
        num_correct = [np.zeros([num_classes, num_clusters]) for _ in range(num_cluster_repeat)]
        num_instance = [np.zeros([num_classes, num_clusters]) for _ in range(num_cluster_repeat)]
        g_corrects = [0 for _ in rebias_model.g_nets]

        if outer_criterion.__class__.__name__ in ['LearnedMixin', 'RUBi']:
            """For computing HSIC loss only.
            """
            outer_criterion = None

        outer_loss = [0 for _ in rebias_model.g_nets]
        inner_loss = [0 for _ in rebias_model.g_nets]

        for x, labels, bias_labels in dataloader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            for bias_label in bias_labels:
                bias_label.to(self.device)

            f_pred, g_preds, f_feat, g_feats = rebias_model(x)

            batch_size = labels.size(0)
            total += batch_size

            if key == 'unbiased':
                num_correct, num_instance = self.imagenet_unbiased_accuracy(f_pred.data, labels, bias_labels,
                                                                            num_correct, num_instance, num_cluster_repeat)
            else:
                f_correct += n_correct(f_pred, labels)
                for idx, g_pred in enumerate(g_preds):
                    g_corrects[idx] += n_correct(g_pred, labels)

            if outer_criterion:
                for idx, g_pred in enumerate(g_preds):
                    outer_loss[idx] += batch_size * outer_criterion(f_pred, g_pred).item()

            if inner_criterion:
                for idx, g_pred in enumerate(g_preds):
                    inner_loss[idx] += batch_size * inner_criterion(f_pred, g_pred).item()

        if key == 'unbiased':
            for k in range(num_cluster_repeat):
                x, y = [], []
                _num_correct, _num_instance = num_correct[k].flatten(), num_instance[k].flatten()
                for i in range(_num_correct.shape[0]):
                    __num_correct, __num_instance = _num_correct[i], _num_instance[i]
                    if __num_instance >= 10:
                        x.append(__num_instance)
                        y.append(__num_correct / __num_instance)
                f_correct += sum(y) / len(x)

            ret = {'f_acc': f_correct / num_cluster_repeat}
        else:
            ret = {'f_acc': f_correct / total}

        for idx, (_g_correct, _outer_loss, _inner_loss) in enumerate(zip(g_corrects, outer_loss, inner_loss)):
            ret['g_{}_acc'.format(idx)] = _g_correct / total
            ret['outer_{}_loss'.format(idx)] = _outer_loss / total
            ret['inner_{}_loss'.format(idx)] = _inner_loss / total
        return ret


class ActionEvaluator(EvaluatorBase):
    @torch.no_grad()
    def evaluate_rebias(self, dataloader, rebias_model,
                        outer_criterion=None,
                        inner_criterion=None,
                        num_classes=50,
                        **kwargs):
        rebias_model.eval()

        num_clips = dataloader.dataset._num_clips
        num_videos = len(dataloader.dataset) // num_clips
        video_f_preds = torch.zeros((num_videos, num_classes))
        video_g_preds = torch.zeros((len(rebias_model.g_nets), num_videos, num_classes))
        video_labels = torch.zeros((num_videos)).long()
        clip_count = torch.zeros((num_videos)).long()

        total = 0

        if outer_criterion.__class__.__name__ in ['LearnedMixin', 'RUBi']:
            """For computing HSIC loss only.
            """
            outer_criterion = None

        outer_loss = [0 for _ in rebias_model.g_nets]
        inner_loss = [0 for _ in rebias_model.g_nets]
        for x, labels, index in dataloader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            f_pred, g_preds, f_feat, g_feats = rebias_model(x)

            for ind in range(f_pred.shape[0]):
                vid_id = int(index[ind]) // num_clips
                video_labels[vid_id] = labels[ind].detach().cpu()
                video_f_preds[vid_id] += f_pred[ind].detach().cpu()
                for g_idx, g_pred in enumerate(g_preds):
                    video_g_preds[g_idx, vid_id] += g_pred[ind].detach().cpu()
                clip_count[vid_id] += 1

            batch_size = labels.size(0)
            total += batch_size

            if outer_criterion:
                for idx, g_pred in enumerate(g_preds):
                    outer_loss[idx] += batch_size * outer_criterion(f_pred, g_pred).item()

            if inner_criterion:
                for idx, g_pred in enumerate(g_preds):
                    inner_loss[idx] += batch_size * inner_criterion(f_pred, g_pred).item()

        if not all(clip_count == num_clips):
            print(
                "clip count {} ~= num clips {}".format(
                    clip_count, num_clips
                )
            )

        f_correct = n_correct(video_f_preds, video_labels)
        g_corrects = [n_correct(video_g_pred, video_labels)
                      for video_g_pred in video_g_preds]

        ret = {'f_acc': f_correct / num_videos}
        for idx, (_g_correct, _outer_loss, _inner_loss) in enumerate(zip(g_corrects, outer_loss, inner_loss)):
            ret['g_{}_acc'.format(idx)] = _g_correct / num_videos

            ret['outer_{}_loss'.format(idx)] = _outer_loss / total
            ret['inner_{}_loss'.format(idx)] = _inner_loss / total
        return ret
