# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup # type: ignore
from timm.utils import accuracy # type: ignore
from typing import Iterable, Optional
import util.misc as misc # type: ignore
import util.lr_sched as lr_sched # type: ignore
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix, precision_score, recall_score, auc, precision_recall_curve, confusion_matrix, cohen_kappa_score # type: ignore
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error # type: ignore
from scipy.stats import pearsonr # type: ignore
from pycm import * # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from util.focal_loss import FocalLoss2d # type: ignore
from util.WeightedLabelSmoothingCrossEntropy import WeightedLabelSmoothingCrossEntropy # type: ignore

def multi_label_target_to_multi_task_target(target,):
    num_classes = target.shape[1]
    target_multi_task = torch.zeros(target.shape[0], num_classes-1, 2, device=target.device, dtype=target.dtype)
    for i in range(num_classes-1):
        target_multi_task[:, i, 0] = target[:, 0]
        target_multi_task[:, i, 1] = target[:, i+1]
    weight_multi_task = target_multi_task.sum(dim=2)
    return target_multi_task, weight_multi_task

def multi_task_loss(output, target, criterion, multi_task_type='multi_task_default'):
    target_multi_task, weight_multi_task = multi_label_target_to_multi_task_target(target)
    num_classes = target.shape[1]
    num_multi_task_classes = num_classes - 1
    loss = 0
    if multi_task_type == 'multi_task_default':
        output = output.reshape(output.shape[0], -1, 2)
        assert output.shape[1] == num_multi_task_classes

    else:
        pass

    for i in range(num_multi_task_classes):
        if multi_task_type == 'multi_task_default':

            output_i = output[:, i]

        else:
            output_i = output[:, [0, i+1]]
        target_i = target_multi_task[:, i]
        target_i = target_i.long()
        weight_i = weight_multi_task[:, i:i+1] # keep the same shape as target_i
        loss += criterion(output_i, target_i)

    loss = loss / (weight_multi_task.sum() + 1e-8)
    return loss

def classwise_accuracy(y_true, y_pred, threshold=0.5):
    """
    Calculate classwise accuracy
    """
    accuracies = []
    for label_class in range(y_true.shape[1]):
        correct = 0
        for i in range(y_true.shape[0]):
            if y_true[i, label_class] == (y_pred[i, label_class] > threshold):
                correct += 1
        accuracies.append(correct / y_true.shape[0])
    return np.array(accuracies)


def misc_measures_multi_task(y_true, y_pred, threshold=0.5, multi_task_type='multi_task_default'):
    print('Going to multi_task function')
    num_classes = y_true.shape[1]
    num_tasks = num_classes - 1
    num_samples = y_true.shape[0]
    task_y_true_list = torch.zeros(num_samples, num_tasks, 2).long().numpy()
    task_y_mask_list = torch.zeros(num_samples, num_tasks).long().numpy()
    task_y_pred_list = torch.zeros(num_samples, num_tasks, 2)

    for i, y in enumerate(y_true):
        task_y_true_list[i, :, 0] = y[0]
        task_y_true_list[i, :, 1] = y[1:]
    if multi_task_type == 'multi_task_default':

        y_pred = y_pred.reshape(y_pred.shape[0], -1, 2)
        assert y_pred.shape[1] == num_tasks
        task_y_pred_list = torch.tensor(y_pred).float()

    else:
        for i, y in enumerate(y_pred):
            task_y_pred_list[i, :, 0] = torch.tensor(y[0])
            task_y_pred_list[i, :, 1] = torch.tensor(y[1:])

    # apply softmax to the output last dimension
    task_y_pred_list = nn.Softmax(dim=2)(task_y_pred_list)
    task_y_mask_list[:] = np.sum(task_y_true_list, axis=2)

    taskwise_acc = []
    taskwise_roc = []
    taskwise_auc_pr = []
    taskwise_precision = []
    taskwise_recall = []
    taskwise_f1 = []
    taskwise_max_f1 = []
    taskwise_AP = []
    taskwise_balanced_acc = []
    taskwise_specificity = []
    taskwise_sensitivity = []
    taskwise_mcc = []
    taskwise_G = []
    taskwise_kappa = []
    temp_y_true = []
    temp_y_pred = []

    for i in range(num_tasks):
        y_true_i = task_y_true_list[:, i]
        y_pred_i = task_y_pred_list[:, i]

        mask_i = task_y_mask_list[:, i]
        # filter out the samples using the mask
        y_true_i = y_true_i[mask_i > 0]
        y_pred_i = y_pred_i[mask_i > 0]
        temp_y_true.append(y_true_i)
        temp_y_pred.append(y_pred_i)

        y_pred_i = y_pred_i.numpy()

        tn, fp, fn, tp = confusion_matrix(y_true_i[:, 1], y_pred_i[:, 1] > threshold).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
        G = np.sqrt(sensitivity * specificity)
        balanced_acc = (sensitivity + specificity) / 2

        auc_roc = roc_auc_score(y_true_i, y_pred_i, average='macro')
        AP = average_precision_score(y_true_i, y_pred_i, average='macro')

        kappa = cohen_kappa_score(y_true_i[:, 1], y_pred_i[:, 1] > threshold)
        agg_auprc = 0
        agg_max_f1 = 0
        for j in range(len(y_pred_i[0])):
            pr, re, ths = precision_recall_curve(y_true_i[:, j], y_pred_i[:, j])
            auprc = auc(re, pr)
            agg_auprc += auprc
            max_f1 = 0
            for j in range(len(pr)):
                temp_f1 = 2 * pr[j] * re[j] / (pr[j] + re[j] + 1e-8)
                max_f1 = max(max_f1, temp_f1)
            agg_max_f1 += max_f1
        auprc = agg_auprc / len(y_pred_i[0])
        max_f1 = agg_max_f1 / len(y_pred_i[0])


        taskwise_acc.append(accuracy)
        taskwise_roc.append(auc_roc)
        taskwise_auc_pr.append(auprc)
        taskwise_precision.append(precision)
        taskwise_recall.append(recall)
        taskwise_f1.append(f1)
        taskwise_max_f1.append(max_f1)
        taskwise_AP.append(AP)
        taskwise_balanced_acc.append(balanced_acc)
        taskwise_specificity.append(specificity)
        taskwise_sensitivity.append(sensitivity)
        taskwise_mcc.append(mcc)
        taskwise_G.append(G)
        taskwise_kappa.append(kappa)

    # calculate micro AP using temp_y_true and temp_y_pred
    temp_y_true = np.concatenate(temp_y_true)
    temp_y_pred = np.concatenate(temp_y_pred)
    micro_AP = average_precision_score(temp_y_true, temp_y_pred, average='micro')

    macro_average_acc = np.mean(taskwise_acc)
    macro_average_roc = np.mean(taskwise_roc)
    macro_average_auc_pr = np.mean(taskwise_auc_pr)
    macro_average_precision = np.mean(taskwise_precision)
    macro_average_recall = np.mean(taskwise_recall)
    macro_average_f1 = np.mean(taskwise_f1)
    macro_average_max_f1 = np.mean(taskwise_max_f1)
    macro_average_AP = np.mean(taskwise_AP)
    macro_average_balanced_acc = np.mean(taskwise_balanced_acc)
    macro_average_specificity = np.mean(taskwise_specificity)
    macro_average_sensitivity = np.mean(taskwise_sensitivity)
    macro_average_mcc = np.mean(taskwise_mcc)
    macro_average_G = np.mean(taskwise_G)
    macro_average_kappa = np.mean(taskwise_kappa)

    return_dict = {}
    return_dict['macro'] = dict()
    return_dict['macro']['micro_AP'] = micro_AP
    return_dict['macro']['accuracy'] = macro_average_acc
    return_dict['macro']['roc_auc'] = macro_average_roc
    return_dict['macro']['precision'] = macro_average_precision
    return_dict['macro']['recall'] = macro_average_recall
    return_dict['macro']['f1'] = macro_average_f1
    return_dict['macro']['max_f1'] = macro_average_max_f1
    return_dict['macro']['AP'] = macro_average_AP
    return_dict['macro']['auprc'] = macro_average_auc_pr
    return_dict['macro']['balanced_acc'] = macro_average_balanced_acc
    return_dict['macro']['specificity'] = macro_average_specificity
    return_dict['macro']['sensitivity'] = macro_average_sensitivity
    return_dict['macro']['mcc'] = macro_average_mcc
    return_dict['macro']['G'] = macro_average_G
    return_dict['macro']['kappa'] = macro_average_kappa

    return_dict['classwise'] = dict()
    return_dict['classwise']['accuracy'] = taskwise_acc
    return_dict['classwise']['roc_auc'] = taskwise_roc
    return_dict['classwise']['precision'] = taskwise_precision
    return_dict['classwise']['recall'] = taskwise_recall
    return_dict['classwise']['f1'] = taskwise_f1
    return_dict['classwise']['max_f1'] = taskwise_max_f1
    return_dict['classwise']['AP'] = taskwise_AP
    return_dict['classwise']['auprc'] = taskwise_auc_pr
    return_dict['classwise']['balanced_acc'] = taskwise_balanced_acc
    return_dict['classwise']['specificity'] = taskwise_specificity
    return_dict['classwise']['sensitivity'] = taskwise_sensitivity
    return_dict['classwise']['mcc'] = taskwise_mcc
    return_dict['classwise']['G'] = taskwise_G
    return_dict['classwise']['kappa'] = taskwise_kappa
    print('return_dict:', return_dict)
    return return_dict

def safe_macro_roc_auc(y_true_onehot, y_pred, multi_class='ovr', average='macro'):
    y_true = np.array(y_true_onehot)
    y_pred = np.array(y_pred)

    # filter class
    valid_classes = y_true.sum(axis=0) > 0
    y_true_valid = y_true[:, valid_classes]
    y_pred_valid = y_pred[:, valid_classes]

    try:
        auc = roc_auc_score(
            y_true_valid,
            y_pred_valid,
            multi_class=multi_class,
            average=average
        )
    except ValueError as e:
        print(f"[Warning] ROC AUC error: {e}")
        auc = np.nan

    return auc






def misc_measures_multi_label(y_true, y_pred, threshold=0.5, **kwargs):
    # ** kwargs is used to pass multi_task_type but will not be used in this function
    """
    Calculate classwise accuracy
    """

    classwise_acc = classwise_accuracy(y_true, y_pred, threshold)
    macro_average_acc = np.mean(classwise_acc)
    classwise_roc = roc_auc_score(y_true, y_pred, average=None)
    macro_average_roc = roc_auc_score(y_true, y_pred, average='macro')
    classwise_precision = precision_score(y_true, y_pred > threshold, average=None)
    macro_average_precision = precision_score(y_true, y_pred > threshold, average='macro')
    classwise_recall = recall_score(y_true, y_pred > threshold, average=None)
    macro_average_recall = recall_score(y_true, y_pred > threshold, average='macro')
    classwise_f1 = f1_score(y_true, y_pred > threshold, average=None)
    macro_average_f1 = f1_score(y_true, y_pred > threshold, average='macro')
    classwise_AP = average_precision_score(y_true, y_pred, average=None)
    macro_average_AP = average_precision_score(y_true, y_pred, average='macro')
    micro_average_AP = average_precision_score(y_true, y_pred, average='micro')
    classwise_balanced_acc = []
    for i in range(y_true.shape[1]):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i] > threshold).ravel()
        sensitivity = tp / (tp + fn + 1e-8)  # sensitivity
        specificity = tn / (tn + fp + 1e-8)  # specificity
        classwise_balanced_acc.append((sensitivity + specificity) / 2)
    macro_average_balanced_acc = np.mean(classwise_balanced_acc)
    aucs = []
    max_f1s = []
    for i in range(y_true.shape[1]):

        pr, re, ths = precision_recall_curve(y_true[:,i], y_pred[:,i])

        aucs.append(auc(re, pr))
        max_f1 = 0
        for j in range(len(pr)):
            temp_f1 = 2 * pr[j] * re[j] / (pr[j] + re[j] + 1e-8)
            max_f1 = max(max_f1, temp_f1)
        max_f1s.append(max_f1)
    classwise_auprc = aucs
    avg_classwise_auprc = np.mean(aucs)
    classwise_specificity = []
    classwise_sensitivity = []
    for i in range(y_true.shape[1]):
        tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_pred[:,i] > threshold).ravel()
        classwise_specificity.append(tn/(tn+fp+1e-8))
        classwise_sensitivity.append(tp/(tp+fn+1e-8))
    macro_average_specificity = np.mean(classwise_specificity)
    macro_average_sensitivity = np.mean(classwise_sensitivity)
    classwise_mcc = []
    for i in range(y_true.shape[1]):
        tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_pred[:,i] > threshold).ravel()
        classwise_mcc.append((tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-8))
    macro_average_mcc = np.mean(classwise_mcc)
    classwise_G = np.sqrt(classwise_recall*classwise_specificity)
    macro_average_G = np.mean(classwise_G)
    classwise_kappa = []
    for i in range(y_true.shape[1]):
        classwise_kappa.append(cohen_kappa_score(y_true[:,i], y_pred[:,i] > threshold))
    macro_average_kappa = np.mean(classwise_kappa)
    return_dict = {}
    return_dict['macro'] = dict()
    return_dict['macro']['accuracy'] = macro_average_acc
    return_dict['macro']['roc_auc'] = macro_average_roc
    return_dict['macro']['precision'] = macro_average_precision
    return_dict['macro']['recall'] = macro_average_recall
    return_dict['macro']['f1'] = macro_average_f1
    return_dict['macro']['AP'] = macro_average_AP
    return_dict['macro']['auprc'] = avg_classwise_auprc
    return_dict['macro']['specificity'] = macro_average_specificity
    return_dict['macro']['sensitivity'] = macro_average_sensitivity
    return_dict['macro']['mcc'] = macro_average_mcc
    return_dict['macro']['G'] = macro_average_G
    return_dict['macro']['micro_AP'] = micro_average_AP
    return_dict['macro']['balanced_acc'] = macro_average_balanced_acc
    return_dict['macro']['kappa'] = macro_average_kappa
    return_dict['macro']['max_f1'] = np.mean(max_f1s)

    return_dict['classwise'] = dict()
    return_dict['classwise']['accuracy'] = classwise_acc
    return_dict['classwise']['roc_auc'] = classwise_roc
    return_dict['classwise']['precision'] = classwise_precision
    return_dict['classwise']['recall'] = classwise_recall
    return_dict['classwise']['f1'] = classwise_f1
    return_dict['classwise']['AP'] = classwise_AP
    return_dict['classwise']['auprc'] = classwise_auprc
    return_dict['classwise']['specificity'] = classwise_specificity
    return_dict['classwise']['sensitivity'] = classwise_sensitivity
    return_dict['classwise']['mcc'] = classwise_mcc
    return_dict['classwise']['G'] = classwise_G
    return_dict['classwise']['balanced_acc'] = classwise_balanced_acc
    return_dict['classwise']['kappa'] = classwise_kappa
    return_dict['classwise']['max_f1'] = max_f1s
    return return_dict


def misc_measures(confusion_matrix, start_cls_idx=0):

    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    delta = 1e-8
    balance_accuracy = []

    for i in range(start_cls_idx, confusion_matrix.shape[0]):
        cm1 = confusion_matrix[i]
        acc.append(1. * (cm1[0, 0] + cm1[1, 1]) / (np.sum(cm1) + delta))
        balance_accuracy.append(1. * (cm1[0, 0] / (cm1[0, 0] + cm1[0, 1] + delta) + cm1[1, 1] / (cm1[1, 0] + cm1[1, 1] + delta)) / 2)
        sensitivity_ = 1. * cm1[1, 1] / (cm1[1,0] + cm1[1, 1] + delta)
        sensitivity.append(sensitivity_)
        specificity_ = 1. * cm1[0, 0] / (cm1[0, 1] + cm1[0, 0] + delta)
        specificity.append(specificity_)
        precision_ = 1. * cm1[1, 1] / (cm1[1, 1] + cm1[0, 1] + delta)
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_ * specificity_))
        F1_score_2.append(2 * precision_ * sensitivity_ / (precision_ + sensitivity_ + delta))
        mcc = (cm1[0, 0] * cm1[1, 1] - cm1[0, 1] * cm1[1, 0]) / (np.sqrt((cm1[0, 0] + cm1[0, 1]) * (cm1[0, 0] + cm1[1, 0]) * (cm1[1, 1] + cm1[1, 0]) * (cm1[1, 1] + cm1[0, 1])) + delta)
        mcc_.append(mcc)

    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    balance_accuracy = np.array(balance_accuracy).mean()

    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_, balance_accuracy



def train_one_epoch_dual(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (oct_images, cfp_images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args) # type: ignore

        oct_images = oct_images.to(device, non_blocking=True)
        cfp_images = cfp_images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(oct_data_loader) + epoch, args) # type: ignore

        # Check if the criterion is BCEWithLogitsLoss and convert targets to float if it is
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss) or isinstance(criterion, FocalLoss2d):

            targets = targets.float()
        elif args.task_mode == 'regression':
            targets = targets.float()

        if mixup_fn is not None: #TODO:Debug
            oct_images, targets = mixup_fn(oct_images, targets)

        with torch.cuda.amp.autocast():
            outputs = model(oct_images, cfp_images)

            if args.task_mode.startswith('multi_task') and isinstance(criterion, WeightedLabelSmoothingCrossEntropy):
                loss = multi_task_loss(outputs, targets, criterion, multi_task_type=args.task_mode)
            else:
                loss = criterion(outputs, targets)
            if not args.not_print_logits:
                print('outputs:', outputs.detach().cpu().numpy(), 'targets', targets, 'loss:', loss.item(), 'input_shape:', samples.shape)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            return None
            # sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(oct_data_loader) + epoch) * 1000) # type: ignore
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def init_csv_writer(task, mode):
    if not os.path.exists(task):
        os.makedirs(task)
    results_path = task + 'metrics_{}.csv'.format(mode)
    with open(results_path, mode='a', newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[['acc', 'bal_acc', 'sensitivity','specificity','precision','auc_roc','auc_pr','F1','mcc','loss']]
        for i in data2:
            wf.writerow(i)
    return results_path


@torch.no_grad()
def evaluate_dual(data_loader, model, device, task, epoch, mode, num_class, criterion=torch.nn.CrossEntropyLoss(), task_mode='binary_cls', disease_list=None, return_bal_acc=False, args=None):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    if not os.path.exists(task):
        os.makedirs(task)

    prediction_decode_list = []
    prediction_list = []
    true_label_list = [] # for regression task
    true_label_decode_list = []
    true_label_onehot_list = []

    # For regression task, we'll handle separately
    if task_mode == 'regression':
        regression_metrics = {'pearsonr': [], 'r2': [], 'explained_variance': [], 'mse': [], 'mae': [], 'loss': 0}

    if task_mode == 'multi_label' or task_mode.startswith('multi_task'):
        multi_label_probs = []
        target_list = []
        threshold = 0.5
        if task_mode.startswith('multi_task'):
            measure_func = misc_measures_multi_task
        else:
            measure_func = misc_measures_multi_label

    # switch to evaluation mode
    model.eval()
    if not hasattr(args, 'frame_inference_all'):
        args.frame_inference_all = False

    if args.frame_inference_all:
        patient_id_list = []
        visit_hash_list = []
        embeddings_list = []
        targets_list = []
    for i, batch in enumerate(metric_logger.log_every(data_loader, 1, header)):
        oct_images = batch[0]
        cfp_images = batch[1]
        target = batch[2]

        if args.frame_inference_all:
            all_info = target
            target = all_info[0]
            patient_id = all_info[1]
            visit_hash = all_info[2]
            print('patient_id:', patient_id, 'visit_hash:', visit_hash, 'target:', target)

            sample_num = oct_images.shape[0]
            frame_num = oct_images.shape[1]
            if args.patient_dataset_type.startswith('Center2D'):
                oct_images = oct_images.reshape(-1, oct_images.shape[2], oct_images.shape[3], oct_images.shape[4])


                print('target before:', target.shape)
                target = torch.repeat_interleave(target, frame_num, dim=0)
                print('target after:', target.shape)

            patient_id_list.extend(patient_id)
            visit_hash_list.extend(visit_hash)
            targets_list.extend(target.cpu().detach().numpy())

        # Check if the criterion is BCEWithLogitsLoss and convert targets to float if it is
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss) or isinstance(criterion, FocalLoss2d):
            target = target.float()

        oct_images = oct_images.to(device, non_blocking=True)
        cfp_images = cfp_images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        true_label = F.one_hot(target.to(torch.int64), num_classes=num_class) if (task_mode == 'binary_cls' or task_mode == 'multi_cls') else target
        # compute output
        with torch.cuda.amp.autocast():
            if hasattr(args, 'return_embeddings') and args.return_embeddings:
                output, embeddings = model(oct_images, cfp_images, return_embeddings=args.return_embeddings)
            else:
                output = model(oct_images, cfp_images)

            loss = criterion(output, target)

            if task_mode == 'regression':
                if len(target.shape) > 1:
                    target = target[:, 0]
                    output = output[:, 0]
                prediction_list.extend(output.cpu().detach().numpy().flatten())
                true_label_list.extend(target.cpu().detach().numpy().flatten())

            # For classification tasks (binary, multi-class, etc.)
            else:
                prediction_softmax = nn.Softmax(dim=1)(output)

                _, prediction_decode = torch.max(prediction_softmax, 1)
                _, true_label_decode = torch.max(true_label, 1)

                if args.frame_inference_all:
                    if hasattr(args, 'return_embeddings') and args.return_embeddings:
                        embeddings_list.extend(embeddings.cpu().detach().numpy())
                    prediction_softmax = prediction_softmax.reshape(sample_num, frame_num, -1)
                    if args.patient_dataset_type.startswith('Center2D'):

                        total_size = torch.prod(torch.tensor(target.shape)).item()
                        if total_size // (sample_num * frame_num) > 1:
                            target = target.reshape(sample_num, frame_num, -1)
                        else:
                            target = target.reshape(sample_num, frame_num)



                prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
                true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
                true_label_onehot_list.extend(true_label.cpu().detach().numpy())
                if task_mode == 'binary_cls' or task_mode == 'multi_cls':
                    prediction_list.extend(prediction_softmax.cpu().detach().numpy())
                if task_mode == 'multi_label':
                    multilabel_prob = nn.Sigmoid()(output)
                    target_list.extend(target.cpu().detach().numpy())
                    prediction_list.extend(multilabel_prob.cpu().detach().numpy())

                elif task_mode.startswith('multi_task'):
                    multi_label_probs.append(output.cpu().detach().numpy())
                    target_list.extend(target.cpu().detach().numpy())
                if args.frame_inference_all:
                    print(len(prediction_list), prediction_list[0].shape, len(patient_id_list), len(visit_hash_list), patient_id_list[0], visit_hash_list[0], len(true_label_decode_list), len(true_label_onehot_list))

        batch_size = oct_images.shape[0]
        metric_logger.update(loss=loss.item())

        if task_mode == 'binary_cls' or task_mode == 'multi_cls':
            acc1,_ = accuracy(output, target, topk=(1,2))
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        elif task_mode == 'multi_label':
            multi_label_probs.append(multilabel_prob.cpu().detach().numpy())


    if task_mode == 'regression':
        # Convert to numpy arrays for metric calculation
        prediction_list = np.array(prediction_list)
        true_label_list = np.array(true_label_list)

        # Calculate regression metrics
        pearson_corr = pearsonr(prediction_list, true_label_list)[0]
        R2 = pearson_corr ** 2
        r2 = r2_score(true_label_list, prediction_list)
        explained_variance = explained_variance_score(true_label_list, prediction_list)
        mse = mean_squared_error(true_label_list, prediction_list)
        mae = mean_absolute_error(true_label_list, prediction_list)

        regression_metrics['pearsonr'].append(pearson_corr)
        regression_metrics['r2'].append(r2)
        regression_metrics['explained_variance'].append(explained_variance)
        regression_metrics['mse'].append(mse)
        regression_metrics['mae'].append(mae)
        regression_metrics['R2'] = R2
        regression_metrics['loss'] = metric_logger.loss.avg

        # Log and print regression metrics
        print('Regression Metrics - Pearsonr: {:.4f} R²: {:.4f} ExplainedVariance: {:.4f} MSE: {:.4f} MAE: {:.4f}, R2: {:.4f}, Loss: {:.4f}'.format(
            pearson_corr, r2, explained_variance, mse, mae, R2, metric_logger.loss.avg
        ))

        results_path = os.path.join(task, 'regression_metrics_{}.csv'.format(mode))
        with open(results_path, mode='a', newline='', encoding='utf8') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Pearsonr', 'R²', 'ExplainedVariance', 'MSE', 'MAE', 'R2', 'Loss'])
            # writer.writerow([pearson_corr, r2, explained_variance, mse, mae])
            writer.writerow([f'{pearson_corr:.4f}', f'{r2:.4f}', f'{explained_variance:.4f}', f'{mse:.4f}', f'{mae:.4f}', f'{R2:.4f}', f'{metric_logger.loss.avg:.4f}'])
        print('Regression metrics saved to:', results_path)
        print('Regression metrics:', regression_metrics)

        return dict([key, np.mean(val)] for key, val in regression_metrics.items())

    if args.frame_inference_all:
        print('prediction_list:', len(prediction_list), prediction_list[0].shape, len(patient_id_list), len(visit_hash_list), patient_id_list[0], visit_hash_list[0], len(true_label_decode_list), len(true_label_onehot_list))
        if hasattr(args, 'return_embeddings') and args.return_embeddings:
            print('embeddings_list:', len(embeddings_list), embeddings_list[0].shape)

        with open(os.path.join(task, 'frame_inference_results.pkl'), 'wb') as f:
            pickle.dump([patient_id_list, visit_hash_list, prediction_list, true_label_decode_list, true_label_onehot_list, embeddings_list, targets_list], f)
        print('Saved frame_inference_results.pkl, exiting...')
        exit()

    if task_mode == 'multi_label' or task_mode.startswith('multi_task'):

        multi_label_probs = np.concatenate(multi_label_probs, axis=0)
        target_list = np.array(target_list)

        results_dict = measure_func(target_list, multi_label_probs, threshold=threshold, multi_task_type=args.task_mode)

        metric_logger.synchronize_between_processes()
        # set acc1 to be the same as macro_metrics['accuracy']
        metric_logger.meters['acc1'].update(results_dict['macro']['accuracy'], n=1)

        # Assuming 'results_dict' is the dictionary containing your metrics
        macro_metrics = results_dict['macro']

        print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f}, AP: {:4f}, AUC-pr: {:.4f} F1-score: {:.4f}, Max F1: {:.4f}, Balanced Acc: {:.4f}, Kappa: {:.4f}, MCC: {:.4f}'.format(
            macro_metrics['accuracy'], macro_metrics['roc_auc'], macro_metrics['AP'], macro_metrics['auprc'], macro_metrics['f1'], macro_metrics['max_f1'], macro_metrics['balanced_acc'], macro_metrics['kappa'], macro_metrics['mcc']
        ))

        # Define path for macro average results
        results_path = os.path.join(task, 'macro_metrics_{}.csv'.format(mode))

        # Save macro average metrics to CSV
        with open(results_path, mode='a', newline='', encoding='utf8') as file:
            writer = csv.writer(file)
            # Check if file is empty to write headers
            if file.tell() == 0:
                writer.writerow(['Accuracy', 'ROC AUC', 'Average Precision', 'AUPRC', 'F1 Score', 'Balanced Acc', 'MCC', 'G-Mean', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'Micro AP', 'Kappa', 'Max F1', 'loss'])
            writer.writerow([
                macro_metrics['accuracy'], macro_metrics['roc_auc'], macro_metrics['AP'], macro_metrics['auprc'], macro_metrics['f1'], macro_metrics['balanced_acc'], macro_metrics['mcc'], macro_metrics['G'], macro_metrics['precision'], macro_metrics['recall'], macro_metrics['sensitivity'], macro_metrics['specificity'], macro_metrics['micro_AP'], macro_metrics['kappa'], macro_metrics['max_f1'], metric_logger.loss
            ])

        classwise_metrics = results_dict['classwise']
        num_classes = len(classwise_metrics['accuracy'])  # Assuming all metrics cover the same number of classes

        if task_mode.startswith('multi_task'):
            if args.multi_task_idx is not None:
                disease_list = {i: disease_list[args.multi_task_idx[i]] for i in range(num_classes)}
            else:
                disease_list = {i: disease_list[i+1] for i in range(num_classes)}
        assert len(disease_list) == num_classes


        # Define the headers (metrics) you want to include in each file
        headers = ['Accuracy', 'ROC AUC', 'Average Precision', 'AUPRC', 'F1 Score', 'Balanced Acc', 'MCC', 'G-Mean', 'precision', 'recall', 'specificity', 'sensitivity', 'Max F1', 'Kappa']

        # Iterate through each class and create a file with all metrics for that class
        for class_index in disease_list.keys():
            extra_idx = 1 if task_mode.startswith('multi_task') else 0
            # Construct the filename for the current class
            class_metrics_filename = os.path.join(task, f'class_{class_index + extra_idx}_{disease_list[class_index]}_metrics_{mode}.csv')

            # Open the file for writing
            with open(class_metrics_filename, mode='a', newline='', encoding='utf8') as file:
                writer = csv.writer(file)

                # Write the headers
                writer.writerow(headers)

                # Collect and write the values for each metric for the current class
                row_values = [
                    classwise_metrics['accuracy'][class_index],
                    classwise_metrics['roc_auc'][class_index],
                    classwise_metrics['AP'][class_index],
                    classwise_metrics['auprc'][class_index],
                    classwise_metrics['f1'][class_index],
                    classwise_metrics['balanced_acc'][class_index],
                    classwise_metrics['mcc'][class_index],
                    classwise_metrics['G'][class_index],
                    classwise_metrics['precision'][class_index],
                    classwise_metrics['recall'][class_index],
                    classwise_metrics['specificity'][class_index],
                    classwise_metrics['sensitivity'][class_index],
                    classwise_metrics['max_f1'][class_index],
                    classwise_metrics['kappa'][class_index]
                ]
                writer.writerow(row_values)
        if mode.startswith('test'):
            for i in disease_list.keys():

                extra_idx = 1 if task_mode.startswith('multi_task') else 0
                binarized_labels = (multi_label_probs[:, i] > threshold).astype(int)
                cm1 = ConfusionMatrix(actual_vector=target_list[:, i].astype(int), predict_vector=binarized_labels)
                if not args.not_save_figs:
                    cm1.plot(cmap = plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")

                    plt.savefig(task + f'confusion_matrix_{mode}_{i+extra_idx}_{disease_list[i]}_epoch_{epoch}.jpg', dpi=600, bbox_inches ='tight')
                    plt.clf()
        eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        eval_stats.update(macro_metrics)
        
        if return_bal_acc:
            return eval_stats, macro_metrics['roc_auc'], (macro_metrics['auprc'], macro_metrics['balanced_acc'])
        else:
            return eval_stats, macro_metrics['roc_auc'], macro_metrics['auprc']


    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list, labels=[i for i in range(num_class)])
    acc = accuracy_score(true_label_decode_list, prediction_decode_list)
    _, sensitivity, specificity, precision, G, F1, mcc, balanced_acc = misc_measures(confusion_matrix)

    print(true_label_onehot_list[:20])
    print(prediction_list[:20])
    try:
        auc_roc = roc_auc_score(true_label_onehot_list, prediction_list, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"[Warning] ROC AUC 計算失敗: {e}")
        auc_roc = safe_macro_roc_auc(true_label_onehot_list, prediction_list, multi_class='ovr', average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list, average='macro')
    metric_logger.synchronize_between_processes()

    print('Sklearn Metrics - Acc: {:.4f} Balanced-Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, balanced_acc, auc_roc, auc_pr, F1, mcc))
    macro_metrics = {
        'accuracy': acc,
        'balanced_acc': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': sensitivity,  # recall is the same as sensitivity in binary classification
        'auc_roc': auc_roc,
        'auprc': auc_pr,
        'f1': F1,
        'mcc': mcc,
        'G': G,
        'kappa': cohen_kappa_score(true_label_decode_list, prediction_decode_list),
    }
    results_path = task + 'metrics_{}.csv'.format(mode)
    with open(results_path, mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc, balanced_acc, sensitivity, specificity, precision, auc_roc, auc_pr, F1, mcc, metric_logger.loss]]
        for i in data2:
            wf.writerow(i)


    if mode.startswith('test') and not args.not_save_figs:
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap = plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(task + f'confusion_matrix_{mode}_epoch_{epoch}.jpg', dpi=600, bbox_inches='tight')

    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    eval_stats.update(macro_metrics)
    if return_bal_acc:
        return eval_stats, auc_roc, (auc_pr, balanced_acc)
    else:
        return eval_stats, auc_roc, auc_pr

