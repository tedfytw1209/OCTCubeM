# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import math
import torch
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pydicom
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc, f1_score, accuracy_score, confusion_matrix

home_directory = os.path.expanduser('~') + '/'
Ophthal_dir = home_directory + 'AI-READI/dataset/'
retClip_directory = home_directory + 'retClip/'
retClip_exp_directory = home_directory + 'retclip_exp/'

disease_name = ['AMD', 'DME', 'POG', 'MH', 'ODR', 'PM', 'CRO', 'RN', 'VD']
model_expr_names = [
    ('your_results', 'MAE3D_nodrop'),
    # ('RETFound-all-path', 'retFound3D'),
    # ('RETFound-center-path', 'retFound2D')
]


model_name_mapping = {'MAE3D_nodrop': 'OCTCube', 'retFound3D': 'RETFound (3D)', 'retFound2D': 'RETFound (2D)'}

def load_patient_list(list_path, split='train', name_suffix='_pat_list.txt'):
    """
    Load the patient list from the list_path
    """
    patient_list = []
    with open(os.path.join(list_path, split+name_suffix), 'r') as f:
        for line in f:
            patient_list.append(line.strip())
    return patient_list

def argparser():
    parser = argparse.ArgumentParser(description='Evaluate the results of retDisease')
    parser.add_argument('--Ophthal_dir', type=str, default=Ophthal_dir, help='Path to the Ophthal directory')
    parser.add_argument('--retClip_directory', type=str, default=retClip_directory, help='Path to the retClip directory')
    parser.add_argument('--retClip_exp_directory', type=str, default=retClip_exp_directory, help='Path to the retClip exp directory')
    parser.add_argument('--retrieval_results_dir', type=str, default=retClip_exp_directory, help='Path to the retrieval results directory')
    parser.add_argument('--result_epoch', type=int, default=50, help='Epoch of the results')
    parser.add_argument('--results', type=str, default='results.pkl', help='Path to the results file')
    parser.add_argument('--output', type=str, default='./retclip_eval_aireadi_laterality/', help='Path to the output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    parser.add_argument('--name', type=str, default='test_query_train_retrieve', help='Name of the dataset')
    parser.add_argument('--split_path', type=str, default=home_directory + 'Oph_cls_task/scr_train_val_test_split_622', help='Path to the split files')
    parser.add_argument('--num_retrieval', type=int, default=300, help='Number of retrieval results to visualize')
    parser.add_argument('--ir_oct', action='store_true', help='IR OCT')
    return parser

def get_ir_visualization(parent_dir, save_dir, query_patient_id, query_visit_id, retrieved_patient_id_list, retrieved_visit_id_list, ranks, title_patient_id=False, model_names=[]):
    """
    Get the IR visualization
    """
    query_patient_id = int(query_patient_id)
    query_dir = parent_dir + query_visit_id
    data = pydicom.dcmread(query_dir)
    print(data)
    query_img = data.pixel_array
    lateral = data.ImageLaterality


    fig, ax = plt.subplots(3, 1 + ranks[0], figsize=(6, 5))
    query_title = str(query_patient_id) + '-\n' + str(lateral) if title_patient_id else ''
    y = 1 if not title_patient_id else -0.3
    fontsize = 12 if not title_patient_id else 5
    print(ranks)



    for i, (retrieved_patient_ids, retrieved_visit_ids, rank, model_name) in enumerate(zip(retrieved_patient_id_list, retrieved_visit_id_list, ranks, model_names)):
        ax[i, 0].imshow(query_img, cmap='gray')
        if i == 0:
            ax[i, 0].set_title(f'Paired IR\n(Ground Truth){query_title}', fontsize=fontsize)
        # ax[i, 0].axis('off')
        ax[i, 0].set_ylabel(f'{model_name_mapping[model_name]}', fontsize=fontsize)
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])

        for j in range(rank):

            retrieved_patient_id = retrieved_patient_ids[j]
            retrieved_visit_id = retrieved_visit_ids[j]
            retrieved_dir = parent_dir + retrieved_visit_id
            retrieved_data = pydicom.dcmread(retrieved_dir)
            retrieved_img = retrieved_data.pixel_array
            retrieved_lateral = retrieved_data.ImageLaterality
            retrieved_patient_id = int(retrieved_patient_id)

            ax[i, j + 1].imshow(retrieved_img, cmap='gray')
            retrieved_title = str(retrieved_patient_id) + '-\n' + str(retrieved_lateral) if title_patient_id else ''
            if i == 0:
                ax[i, j + 1].set_title(f'Top {j+1}\n retrieved', fontsize=fontsize, y=y)
            ax[i, j + 1].axis('off')



    fig.tight_layout()
    plt.show()
    if not os.path.exists(save_dir + str(query_patient_id) + '/'):
        os.makedirs(save_dir + str(query_patient_id) + '/')
    if not os.path.exists(save_dir + str(query_patient_id) + '/' + lateral + '/'):
        os.makedirs(save_dir + str(query_patient_id) + '/' + lateral + '/')
    plt.savefig(save_dir + str(query_patient_id) + '/' + lateral + f'/rank_{rank}_{query_patient_id}_{lateral}.png')
    plt.close()


def get_laterality(visit):
    """
    Get the laterality of the visit
    """
    visit = Ophthal_dir + visit
    data = pydicom.dcmread(visit)
    return data.ImageLaterality


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    args.ir_oct = True
    train_pat_id = load_patient_list(args.split_path, split='train', name_suffix='_pat_list.txt')
    val_pat_id = load_patient_list(args.split_path, split='val', name_suffix='_pat_list.txt')
    test_pat_id = load_patient_list(args.split_path, split='test', name_suffix='_pat_list.txt')
    train_pat_id_set = set(train_pat_id)
    val_pat_id_set = set(val_pat_id)
    test_pat_id_set = set(test_pat_id)

    retrieval_results_list = []

    for expr_name, model_name in model_expr_names:
        retrieval_results_file_path = args.retrieval_results_dir + expr_name + '/checkpoints/' + f'retrieval_results_{args.result_epoch}.pkl'

        with open(retrieval_results_file_path, 'rb') as f:
            retrieval_results = pkl.load(f)

        retrieval_results_list.append((retrieval_results, model_name))

    sorted_patient_id_dict = {}
    train_id = []
    val_id = []
    test_id = []

    # Assuming that the idx order is the same across all results
    retrieval_results, _ = retrieval_results_list[0]
    patient_list_list = [retrieval_results['labels'][i][1] for i in range(len(retrieval_results['labels']))]
    patient_list = [item for sublist in patient_list_list for item in sublist]

    visit_list_list = [retrieval_results['labels'][i][2] for i in range(len(retrieval_results['labels']))]
    visit_list = [item for sublist in visit_list_list for item in sublist]

    print(len(patient_list), len(visit_list))
    laterality_list = [get_laterality(visit) for visit in visit_list]
    laterality_binary = [1 if laterality == 'R' else 0 for laterality in laterality_list]
    laterality_binary = torch.tensor(laterality_binary)
    print(laterality_list, len(laterality_list))
    print(laterality_binary, len(laterality_binary))


    for i, patient_id in enumerate(patient_list):
        if int(patient_id) not in sorted_patient_id_dict:
            sorted_patient_id_dict[int(patient_id)] = []
        sorted_patient_id_dict[int(patient_id)].append(i)
        if patient_id in train_pat_id_set:
            train_id.append(i)
        elif patient_id in val_pat_id_set:
            val_id.append(i)
        elif patient_id in test_pat_id_set:
            test_id.append(i)
    test_id = list(range(len(patient_list)))




    retrieval_logits_list = []

    for retrieval_results, model_name in retrieval_results_list:
        image_features = torch.tensor(retrieval_results['image_features'])
        text_features = torch.tensor(retrieval_results['text_features'])
        logit_scale = torch.tensor(retrieval_results['logit_scale'][0])
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        if args.ir_oct:
            print('IR OCT')
            temp = logits_per_image
            logits_per_image = logits_per_text
            logits_per_text = temp

        logits_per_image_wo_self = logits_per_image - torch.eye(logits_per_image.shape[0]) * 1e5


        for patient_id, indices in sorted_patient_id_dict.items():
            for i in indices:
                logits_per_image_wo_self[i, indices] = -1e5

        retrieval_logits_list.append(logits_per_image_wo_self)

    test_oct_patient_list = [patient_list[i] for i in test_id]
    test_oct_visit_list = [visit_list[i] for i in test_id]


    results_dict = {}
    topk_list = [1, 3, 5, 10]
    for topk in topk_list:
        for logits_per_image_wo_self, (retrieval_results, model_name) in zip(retrieval_logits_list, retrieval_results_list):
            max_indices = torch.argmax(logits_per_image_wo_self, dim=1)
            topk_indices = torch.topk(logits_per_image_wo_self, topk, dim=1)
            _, topk_indices = topk_indices

            topk_laterality = torch.tensor([laterality_binary[i] for i in topk_indices.flatten()]).reshape(topk_indices.shape)
            # print(topk_laterality, topk_laterality.shape)
            matched_laterality = torch.zeros_like(topk_laterality)
            matched_laterality[topk_laterality == laterality_binary.unsqueeze(1)] = 1
            micro_accuracy = torch.sum(matched_laterality) / (topk_laterality.shape[0] * topk_laterality.shape[1])
            macro_accuracy = torch.mean(torch.sum(matched_laterality, dim=1) / topk_laterality.shape[1])
            print(f'{model_name}: Top {topk} Micro Accuracy: {micro_accuracy}, Macro Accuracy: {macro_accuracy}')

            if model_name not in results_dict:
                results_dict[model_name] = {}
            results_dict[model_name][topk] = {'Micro Accuracy': micro_accuracy.item(), 'Macro Accuracy': macro_accuracy.item()}
    for model_name, topk_dict in results_dict.items():
        print(model_name)
        for topk, acc_dict in topk_dict.items():
            print(f'Top {topk}: {acc_dict}')
        # save the results
        results_df = pd.DataFrame(results_dict[model_name])
        if args.ir_oct:
            results_df.to_csv(args.output + model_name + '_laterality_ir' + '.csv', index=False)
        else:
            results_df.to_csv(args.output + model_name + '_laterality' + '.csv', index=False)



    for i in range(args.num_retrieval):
        retrieved_patient_id_list = []
        retrieved_visit_id_list = []
        ranks = []
        model_names = []

        for logits_per_image_wo_self, (retrieval_results, model_name) in zip(retrieval_logits_list, retrieval_results_list):
            max_indices = torch.argmax(logits_per_image_wo_self, dim=1)
            topk_indices = torch.topk(logits_per_image_wo_self, topk, dim=1)
            _, topk_indices = topk_indices

            topk_patient = [patient_list[topk_indices[test_id[i]][j]] for j in range(topk)]
            topk_visit = [visit_list[topk_indices[test_id[i]][j]] for j in range(topk)]

            topk_laterality = [laterality_list[topk_indices[test_id[i]][j]] for j in range(topk)]

            retrieved_patient_id_list.append(topk_patient)
            retrieved_visit_id_list.append(topk_visit)

            ranks.append(topk)  # Assuming rank 1 for simplicity
            model_names.append(model_name)


        if not os.path.exists(args.output + args.name + '/visualization/'):
            os.makedirs(args.output + args.name + '/visualization/')
        get_ir_visualization(Ophthal_dir, args.output + args.name + '/visualization/', test_oct_patient_list[i], test_oct_visit_list[i], retrieved_patient_id_list, retrieved_visit_id_list, ranks, model_names=model_names)
