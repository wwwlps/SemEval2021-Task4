import os
import argparse
import sys
from config import args as default_args  # 导入config里的默认参数
from config import project_root_path, Baseline_model
from utils.trainer import Trainer
from processor.ReCAM import *
from transformers import (
    RobertaTokenizer,
    AlbertConfig,
    ElectraTokenizer,
    AlbertTokenizer,
    BartConfig,
)
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
from model.baseline import ElectraForMultipleChoice, AlbertForMultipleChoice, \
    RobertaForMultipleChoice, BertForMultipleChoice, ElectraForMultipleChoiceHard, ElectraForMultipleChoiceBinary, ElectraForMultipleChoicePro
from model.GAT_goal import GAT_goal_model
from model.multi_task import ElectraForMultipleChoice_Mul
from model.DCMN_goal import ElectraForMultipleChoiceWithMatch
from processor.ReCAM_DCMN import load_and_cache_examples_DCMN
from model.SVM import ElectraForMultipleChoiceSVM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=default_args['batch_size'], metavar='N',
                        help='input batch size for training (default: {})'.format(default_args['batch_size']))
    parser.add_argument('--test-batch-size', type=int, default=default_args['test_batch_size'], metavar='N',
                        help='input batch size for testing (default: {})'.format(default_args['test_batch_size']))
    parser.add_argument('--epochs', type=int, default=default_args['epochs'], metavar='N',
                        help='number of epochs to train (default: {})'.format(default_args['epochs']))
    parser.add_argument('--fine-tune-epochs', type=int, default=default_args['fine_tune_epochs'], metavar='N',
                        help='number of fine-tune epochs to train (default: {})'.format(
                            default_args['fine_tune_epochs']))
    parser.add_argument('--lr', type=float, default=default_args['lr'], metavar='LR',
                        help='learning rate (default: {})'.format(default_args['lr']))
    parser.add_argument('--fine-tune-lr', type=float, default=default_args['fine_tune_lr'], metavar='LR',
                        help='fine-tune learning rate (default: {})'.format(default_args['fine_tune_lr']))
    parser.add_argument('--adam-epsilon', type=float, default=default_args['adam_epsilon'], metavar='M',
                        help='Adam epsilon (default: {})'.format(default_args['adam_epsilon']))
    parser.add_argument('--max-seq-length', type=int, default=default_args['max_seq_length'], metavar='N',
                        help='max length of sentences (default: {})'.format(default_args['max_seq_length']))
    parser.add_argument('--accumulation-steps', type=int, default=default_args['accumulation_steps'], metavar='N',
                        help='accumulation_steps (default: {})'.format(default_args['accumulation_steps']))
    parser.add_argument('--unfixed-layer', type=int, default=default_args['unfixed_layer'], metavar='N',
                        help='number of unfixed layers (default: {})'.format(default_args['unfixed_layer']))
    parser.add_argument('--subtask-id', type=str, default=default_args['subtask_id'], required=False,
                        choices=['1', '2'],
                        help='subtask 1 or 2 (default: {})'.format(default_args['subtask_id']))
    parser.add_argument('--with-projection-plm', action='store_true', default=False,
                        help='Add non-linear activation before classifier for PLM model')
    parser.add_argument('--with-projection-gat', action='store_true', default=False,
                        help='Add non-linear activation before classifier for GAT model')
    parser.add_argument('--with-kemb', action='store_true', default=False,
                        help='Add Knowledge-enhanced Embedding (KEmb)')
    parser.add_argument('--with-kegat', action='store_true', default=False,
                        help='Add Knowledge-enhanced Graph Attention Network (KEGAT)')
    parser.add_argument('--with-MstageMtask', action='store_true', default=False,
                        help='finetune with NLI task first and do Multitask with RACE')
    parser.add_argument('--with-DCMN', action='store_true', default=False, help='DCMN methods')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()  # 获取用户输入的参数，如果不输入，就是默认使用config文件里的默认参数，

    # 如果觉得输入参数麻烦，可以和之前一样，在config文件里面修改参数，然后直接python main.py

    torch.manual_seed(args.seed)

    for key in default_args.keys():
        if hasattr(args, key):  # 将输入的参数更新至 default_args (如果你输入了参数的话)
            default_args[key] = getattr(args, key)

    args = default_args
    print(args)
    print(project_root_path)
    os.chdir(project_root_path)
    print(args['pretrained_model_path'])  # 参数里的预训练模型文件路径大家可以在config文件里改成自己的

    # pretrained_model_path = args['pretrained_model_path'] + 'roberta-large'  # 预训练语言模型文件目录
    # model_name = 'roberta-large'  # 模型名字，能区分就行
    # tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)

    pretrained_model_path = args['pretrained_model_path'] + 'Electra-large'  # 预训练语言模型文件目录
    model_name = 'Electra-large'  # 模型名字，能区分就行
    tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_path)
    if args['with_kegat']:  # 是否使用GAT模型（默认在electra基础上）
        model_name = 'electra-kegat'
        model = GAT_goal_model(args, is_add_projection=args['with_projection_gat'])
        # model.to(args['device'])
        # model.to(args['device'])
        # best_model_temp_path = '/mnt/ssd/wwwlps/Task4/checkpoints1/best_model_91.87electra-kegat_2021.01.22-15.10.11.pth'
        # if os.path.isfile(best_model_temp_path):
        #     print('load best model: {}'.format(best_model_temp_path))
        #     # 加载最优一次的权重
        #     model.load_state_dict(torch.load(
        #         best_model_temp_path
        #     ))
    elif args['with_MstageMtask']:
        model_name = 'Electra-large'
        model_name_after = 'Electra-large-with_MstageMtask'
        model = ElectraForMultipleChoice_Mul.from_pretrained(pretrained_model_path, task_output_config=[5, 4],
                                                             do_not_summary=False,
                                                             same_linear_layer=True)
    elif args['with_DCMN']:
        model_name = 'Electra-large-with_DCMN'
        model = ElectraForMultipleChoiceWithMatch.from_pretrained(pretrained_model_path)

    else:  # 裸预训练模型，大家可修改为Bert，Roberta，Electra，Albert等
        model_name = 'Electra-large'
        if args['with_kemb']:
            model_name = 'Electra-kemb'
        tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_path)
        model = ElectraForMultipleChoice.from_pretrained(pretrained_model_path,
                                                         is_add_projection=args['with_projection_plm'])
        # model_name = 'roberta-large'
        # tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
        # model = RobertaForMultipleChoice.from_pretrained(pretrained_model_path)
        #
        # best_model_temp_path = os.path.join(args['checkpoints_dir'], 'best_model_Electra-large_2021.01.25-01.46.43.pth')
        # model.to(args['device'])
        # # best_model_temp_path = '/mnt/ssd/wwwlps/Task4/checkpoints/best_model_91.89Electra-large_2021.01.25-01.46.43.pth'
        # best_model_temp_path = '/mnt/ssd/wwwlps/Task4/checkpoints1/best_model_92.35Electra-large_2021.01.21-22.04.28.pth'
        # if os.path.isfile(best_model_temp_path):
        #     print('load best model: {}'.format(best_model_temp_path))
        #     # 加载最优一次的权重
        #     model.load_state_dict(torch.load(
        #         best_model_temp_path
        #     ))

    if args['with_DCMN']:
        train_features = load_and_cache_examples_DCMN(args, model_name, tokenizer, set_type='train')
        dev_features = load_and_cache_examples_DCMN(args, model_name, tokenizer, set_type='dev')
        train_data = Trainer.create_datasets(train_features,
                                             shuffle=True,
                                             batch_size=args['batch_size'])
        dev_data = Trainer.create_datasets(dev_features,
                                           shuffle=False,
                                           batch_size=args['test_batch_size'])
    else:
        train_features = load_and_cache_examples(args, model_name, tokenizer, set_type='train')
        dev_features = load_and_cache_examples(args, model_name, tokenizer, set_type='dev')
        # test_features = load_and_cache_examples(args, model_name, tokenizer, set_type='test')
        train_data = Trainer.create_datasets(train_features,
                                             shuffle=True,
                                             batch_size=args['batch_size'])
        dev_data = Trainer.create_datasets(dev_features,
                                           shuffle=False,
                                           batch_size=args['test_batch_size'])
        # test_data = Trainer.create_datasets(test_features,
        #                                     shuffle=False,
        #                                     batch_size=args['test_batch_size'])
    if args['with_MstageMtask']:
        RACE_train_features = load_and_cache_examples_from_RACE(args, model_name_after, tokenizer, set_type='train',
                                                                level='middle', proportion=5)
        RACE_train_data = Trainer.create_datasets(RACE_train_features,
                                                  shuffle=True,
                                                  batch_size=args['batch_size'])
        train_datasets = [train_data, RACE_train_data]
        print(len(train_data))
        print(len(RACE_train_data))
        # model.to(args['device'])
        # # 加载Multistage阶段后的模型参数
        # best_model_temp_path = os.path.join(args['checkpoints_dir'], 'best_model_temp_2021.01.03-16.21.43.pth')
        # if os.path.isfile(best_model_temp_path):
        #     print('load best model: {}'.format(best_model_temp_path))
        #     # 加载最优一次的权重
        #     model_dict = model.state_dict()  # 现在的模型
        #     pretrained_dict = torch.load(best_model_temp_path)  # 已经预训练好的模型
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                        k in model_dict}  # filter out unnecessary keys
        #
        #     model_dict.update(pretrained_dict)
        #     model.load_state_dict(model_dict)

        model.to(args['device'])
        best_model_temp_path = os.path.join(args['checkpoints_dir'],
                                            'best_model_Electra-large-with_MstageMtask_2021.01.03-17.53.37.pth')
        if os.path.isfile(best_model_temp_path):
            print('load best model: {}'.format(best_model_temp_path))
            # 加载最优一次的权重
            model.load_state_dict(torch.load(
                best_model_temp_path
            ))

    if args['with_MstageMtask']:
        acc, _ = Trainer.multi_train_and_finetune(model, model_name_after, train_datasets, dev_data, args,
                                                  subtask=int(args['subtask_id']))
        # acc, _ = Trainer.train_and_finetune(model, model_name, train_data, train_data, dev_data, args,
        #                                     subtask=int(args['subtask_id']))
    else:
        acc, _ = Trainer.train_and_finetune(model, model_name, train_data, train_data, dev_data, args,
                                            subtask=int(args['subtask_id']))
        # model.to(args['device'])
        # # best_model_temp_path = os.path.join(args['checkpoints_dir'], 'best_trial_90.85_Electra-large_2021.01.07-10.54.55.pth')
        # best_model_temp_path = '/mnt/ssd/wwwlps/Task4/logs1/Electra_hard_2021.01.08-10.53.59'
        # if os.path.isfile(best_model_temp_path):
        #     print('load best model: {}'.format(best_model_temp_path))
        #     # 加载最优一次的权重
        #     model.load_state_dict(torch.load(
        #         best_model_temp_path
        #     ))

        # test_loss, test_acc, test_pred, pred_error_list, pred_label_list, real_label_list = Trainer.test_only(model, test_data, args)
        # print(np.sum(pred_label_list == real_label_list))
        # id_list = np.arange(len(pred_label_list))
        # data_all = pd.DataFrame(np.stack((id_list, pred_label_list)).T, columns=['id', 'label'])
        # data_all.to_csv(
        #     os.path.join('Answer/', 'subtask2_with_model1_91.85.csv'),
        #     columns=['id', 'label'],
        #     header=False,
        #     index=False
        # )
        # test_loss, test_acc, test_pred, pred_error_list, pred_label_list, real_label_list = Trainer.test_only(model,
        #                                                                                                       dev_data,
        #                                                                                                       args)
        # A1 = np.load('datasets/Task1/Four_I.npz')['arr_0']
        # A2 = np.load('datasets/Task1/Four_I.npz')['arr_1']
        # A3 = np.load('datasets/Task1/Four_I.npz')['arr_2']
        # A4 = np.load('datasets/Task1/Four_I.npz')['arr_3']
        #
        # B1 = np.load('datasets/Task1/Ave_I.npz')['arr_0']
        # B2 = np.load('datasets/Task1/Ave_I.npz')['arr_1']
        # B3 = np.load('datasets/Task1/Ave_I.npz')['arr_2']
        # B4 = np.load('datasets/Task1/Ave_I.npz')['arr_3']
        #
        # acc_A1 = np.sum(pred_label_list[A1] == real_label_list[A1]) / len(A1)
        # acc_A2 = np.sum(pred_label_list[A2] == real_label_list[A2]) / len(A2)
        # acc_A3 = np.sum(pred_label_list[A3] == real_label_list[A3]) / len(A3)
        # acc_A4 = np.sum(pred_label_list[A4] == real_label_list[A4]) / len(A4)
        #
        # acc_B1 = np.sum(pred_label_list[B1] == real_label_list[B1]) / len(B1)
        # acc_B2 = np.sum(pred_label_list[B2] == real_label_list[B2]) / len(B2)
        # acc_B3 = np.sum(pred_label_list[B3] == real_label_list[B3]) / len(B3)
        # acc_B4 = np.sum(pred_label_list[B4] == real_label_list[B4]) / len(B4)
        #
        # print(len(A1), len(A2), len(A3), len(A4))
        # print(acc_A1, acc_A2, acc_A3, acc_A4)
        # print(acc_B1, acc_B2, acc_B3, acc_B4)
        # print(np.sum(pred_label_list == real_label_list))
        # id_list = np.arange(len(pred_label_list))
        # data_all2 = pd.DataFrame(np.stack((id_list, real_label_list)).T, columns=['id', 'label'])
        # data_all2.to_csv(
        #     os.path.join('Answer/', 'subtask2_real.csv'),
        #     columns=['id', 'label'],
        #     header=False,
        #     index=False
        # )
        # print(pred_error_list)
        # train_feature, train_label = Trainer.test_SVM(model, train_data, args)
        # test_feature, test_label = Trainer.test_SVM(model, dev_data, args)
        # print(train_feature.shape)
        # print(train_label.shape)
        # print(test_feature.shape)
        # print(test_label.shape)
        # train_real_label = np.argmax(train_label.reshape(-1, 5), axis=1)
        # test_real_label = np.argmax(test_label.reshape(-1, 5), axis=1)
        # print(test_real_label.shape)

        # train_feature = np.load('datasets/train_feature.npy')
        # train_label = np.load('datasets/train_label.npy')
        # test_feature = np.load('datasets/test_feature.npy')
        # test_label = np.load('datasets/test_label.npy')
        # train_real_label = np.argmax(train_label.reshape(-1, 5), axis=1)
        # test_real_label = np.argmax(test_label.reshape(-1, 5), axis=1)
        #
        # parameters = [
        #     {
        #         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        #         'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        #         'kernel': ['rbf']
        #     },
        #     {
        #         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        #         'kernel': ['linear']
        #     },
        #     {
        #         'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        #         'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        #         'kernel': ['poly']
        #     }
        # ]
        # svc = svm.SVC(probability=True)
        # clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)
        # clf.fit(train_feature, train_label)
        #
        # best_model = clf.best_estimator_
        #
        # P = best_model.predict(test_feature)
        # P = P.reshape(-1, 5)
        # print(P[:4])
        # P_score = best_model.predict_proba(test_feature)
        # P_score = P_score.reshape(-1, 5, 2)
        # print(P_score[:4])
        # acc = np.sum(np.argmax(P_score[:, :, 1], axis=1) == test_real_label) / len(test_real_label)
        # print('acc: ', acc)
    print('acc: ', acc)
