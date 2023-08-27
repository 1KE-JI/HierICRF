import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        if input.dim() == 3:
            input = input.view(-1, input.shape[-1])
            target = target.flatten()
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss


class FocalLoss3(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss3, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 1e-6

    def forward(self, logits, labels, device="cuda:0"):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if logits.dim() == 3:
            if logits.shape[-2] == labels.shape[-1]:
                logits = logits.transpose(2, 1)

        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1).to(device)
        tmp = torch.zeros([batch_size, labels_length, seq_length]).to(device)
        label_onehot = tmp.scatter_(1, new_label, 1).to(device)
        # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        fl = -self.alpha * (1 - pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()