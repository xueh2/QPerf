
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

"""
These utility functions are duplicated mainly from 
https://github.com/ternaus/robot-surgery-segmentation

for the implementation of jaccard index

"""

def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x

def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def check_accuracy_test(loader, model, criterion, device=torch.device('cpu'), dtype=torch.float): 
    num_correct = 0
    num_samples = 0
    running_loss = 0.0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            if device is None:
                x = x.to(dtype).cuda() 
                y = y.to(torch.long).cuda()
            else:
                x = x.to(device=device, dtype=dtype) 
                y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = criterion(scores, y)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            running_loss += loss.item() * x.shape[0]
        acc = float(num_correct) / num_samples
        loss = running_loss/ num_samples
        # print('\033[1m' + 'Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    return acc, loss

def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix

def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices

class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        #self.nll_loss = nn.BCEWithLogitsLoss()
        self.nll_loss = nn.BCELoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        
        probs = torch.sigmoid(outputs)        
        loss = (1 - self.jaccard_weight) * self.nll_loss(probs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = probs

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = cuda(torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None

        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        
        log_prob = torch.log_softmax(outputs, dim=1)
        loss = (1 - self.jaccard_weight) * self.nll_loss(log_prob, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = log_prob[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss
