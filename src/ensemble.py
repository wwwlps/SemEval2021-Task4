import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from utils.my_dataset import MyDataset, MyDataLoader, InfiniteDataLoader
from transformers import RobertaTokenizer, RobertaConfig
from transformers import AlbertTokenizer, AlbertConfig
from transformers import BertTokenizer, BertConfig
from transformers import ElectraTokenizer, ElectraConfig
from model.baseline import ElectraForMultipleChoice, AlbertForMultipleChoice, \
    RobertaForMultipleChoice, BertForMultipleChoice, ElectraForMultipleChoiceHard, ElectraForMultipleChoicePro
from model.GAT_goal import GAT_goal_model
import torch.optim as optim
from utils.functions import gelu
from torch.nn import CrossEntropyLoss

from utils.trainer import Trainer


# %%  存储
def save_graph_pickle(res, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(res, f)


# %%  加载
def load_graph_pickle(fpath):
    graph_zip = None
    with open(fpath, 'rb') as f:
        graph_zip = pickle.load(f)
    return graph_zip


# %%
class StackingNNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(StackingNNet, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x, labels=None):
        x = self.fc1(x)
        outputs = gelu(x),
        # outputs = torch.relu(x),
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs[0], labels)

            outputs = (loss,) + outputs
        return outputs


# %%
class Stacking:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.train_features = None
        self.test_features = None
        self.data_random_perm = None
        # self.all_model_name = ['Electra1']
        # self.all_model_name = ['Electra91.3', 'Electra_GAT_RELU_91.8', 'Roberta']
        # self.all_model_name = ['Electra_trail', 'Electra_GAT_RELU_91.8', 'Roberta']
        # self.all_model_name = ['Electra_GAT_RELU', 'Electra_hard', 'Roberta']
        # self.all_model_name = ['Electra', 'Electra_hard', 'Roberta']
        # self.all_model_name = ['Electra_trail']
        # self.all_model_name = ['Roberta1', 'Roberta2']
        # self.all_model_name = ['Electra_Pro']
        # self.all_model_name = ['Electra91.3']
        # self.all_model_name = ['Electra_trail_Electra_GAT_RELU', 'Electra_hard', 'Roberta']
        # self.all_model_name = ['Electra_trail_Electra_GAT_RELU', 'Electra_hard', 'Roberta1_Roberta2']
        # self.all_model_name = ['Electra1', 'Electra_GAT_RELU_Electra_Pro', 'Roberta']
        # self.all_model_name = ['Electra_GAT_RELU_91.8']
        # self.all_model_name = ['Electra5', 'Electra_GAT_RELU91.8', 'Electra1', 'Roberta']
        self.all_model_name2 = ['Electra_GAT_RELU91.8', 'Electra1', 'Roberta']
        self.all_model_name = ['Electra5', 'Electra1', 'Electra_GAT_RELU91.8', 'Roberta']
        # self.all_model_name = ['Electra_GAT_RELU91.8', 'Electra1', 'Roberta']
        # self.all_model_name = ['Electra_GAT_RELU', 'Electra_Pro']
        # self.all_model_name = ['Electra_Electra_Pro', 'Electra_GAT']
        # self.all_model_name = ['Electra_Electra_Pro', 'Electra_GAT']
        # self.all_model_name = ['Roberta']

    def train(self, model, train_data, optimizer):
        model.train()
        pbar = tqdm(train_data, ncols=80)
        # correct 代表累计正确率，count 代表目前已处理的数据个数
        correct = 0
        count = 0
        train_loss = 0.0
        pred_list = []
        accumulation_steps = self.args['accumulation_steps']
        for step, (x, y) in enumerate(pbar):
            y = y.to(self.args['device'])
            output = model(x, labels=y)
            loss = output[0].mean()
            loss = loss / accumulation_steps
            loss.backward()
            if ((step + 1) % accumulation_steps) == 0:
                # 使用了梯度累积，实际batch_size等于args['batch_size']*accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
            # 得到预测结果
            pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
            pred_list.append(output[1].softmax(dim=1))
            # print(output[1].softmax(dim=1))
            # 计算正确个数
            correct += pred.eq(y.view_as(pred)).sum().item()
            count += len(x)
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': '{:.3f}'.format(loss.item()),
                'acc': '{:.3f}'.format(correct * 1.0 / count)
            })
            # gpu_track.track()
        pbar.close()
        return train_loss / count, correct * 1.0 / count, torch.cat(pred_list, dim=0)

    def test(self, model, test_data):
        model.eval()
        test_loss = 0
        correct = 0
        count = 0
        pred_list = []
        with torch.no_grad():
            for step, (x, y) in enumerate(test_data):
                x, y = x.to(self.args['device']), y.to(self.args['device'])
                output = model(x, labels=y)
                loss = output[0]
                test_loss += loss.item()
                pred = output[1].softmax(dim=1).argmax(dim=1, keepdim=True)
                pred_list.append(output[1].softmax(dim=1))
                correct += pred.eq(y.view_as(pred)).sum().item()
                count += len(x)

        test_loss /= count
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                test_loss, correct, count, 100. * correct / count))
        return test_loss, correct * 1.0 / count, torch.cat(pred_list, dim=0)

    def get_save_path(self):  # ensemble后的模型路径
        """
        获取模型保存的路径
        路径地址：./logs/Roberta_Bert_2020.02.21-18.03.52
        :return:
        """
        prefix_path = '../logs_test22'
        # prefix_path = '../logs'
        folder_name = '{}_{}'.format('_'.join(self.all_model_name), self.args['exec_time'])
        path = os.path.join(prefix_path, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_save_path2(self):  # ensemble后的模型路径
        """
        获取模型保存的路径
        路径地址：./logs/Roberta_Bert_2020.02.21-18.03.52
        :return:
        """
        prefix_path = '../logs_test22'
        # prefix_path = '../logs'
        folder_name = '{}_{}'.format('_'.join(self.all_model_name2), self.args['exec_time'])
        path = os.path.join(prefix_path, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def start_train(self, train_data, test_data):
        stacking_model = StackingNNet(train_data[0][0].size(1), train_data[0][0].size(1) // len(self.all_model_name))
        stacking_model.to(self.args['device'])
        optimizer = optim.Adam(stacking_model.parameters(), lr=self.args['ensemble_lr'])

        acc = 0
        for epoch in range(self.args['ensemble_epochs']):
            print('Epoch {}/{}'.format(epoch + 1, self.args['ensemble_epochs']))
            train_loss, train_acc, train_pred = self.train(stacking_model, train_data, optimizer)
            test_loss, test_acc, test_pred = self.test(stacking_model, test_data)
            if test_acc > acc:
                torch.save(stacking_model.state_dict(), os.path.join(self.get_save_path(), 'stacking_model.pth'))
            acc = max(acc, test_acc)
        print('final acc: {:.4f}'.format(acc))
        # torch.save(stacking_model.state_dict(),
        #            os.path.join(self.get_save_path(), 'stacking_model.pth'))

    def start_with_ensemble_data(self, ensemble_folder):
        # 先加载要送往 StackingNet 的数据，并传入 device 中
        all_train_pred, all_test_pred = load_graph_pickle(os.path.join(
            ensemble_folder,
            'ensemble_data.pickle',
        ))
        self.start_train(all_train_pred, all_test_pred)

    def start_with_final_pred(self):
        all_train_pred = []
        all_test_pred = []
        for i in self.all_model_name:
            self.init_pre_exec(i)
            # 将各种不同模型的结果整合起来
            final_pred = load_graph_pickle(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(i)))
            all_train_pred.append(final_pred[0])
            all_test_pred.append(final_pred[1])
        all_train_pred = torch.cat(all_train_pred, dim=1)
        all_test_pred = torch.cat(all_test_pred, dim=1)

        train_data = MyDataLoader(MyDataset(all_train_pred,
                                            torch.tensor([i[1] for i in self.train_features])),
                                  batch_size=self.args['batch_size'])
        test_data = MyDataLoader(MyDataset(all_test_pred,
                                           torch.tensor([i[1] for i in self.test_features])),
                                 batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        train_data = list(train_data)
        test_data = list(test_data)

        # 将该次的数据存储在本地，可以用来单独测试
        save_graph_pickle((train_data, test_data),
                          os.path.join(self.get_save_path(), 'ensemble_data.pickle'))
        self.start_train(train_data, test_data)

    def start(self):  # 获取ensemble数据特征
        all_train_pred = []
        all_test_pred = []
        for i in self.all_model_name:
            # 将各种不同模型的结果整合起来
            final_train_pred, final_test_pred = self.single_model_execution(i)
            all_train_pred.append(final_train_pred)
            all_test_pred.append(final_test_pred)

        all_train_pred = torch.cat(all_train_pred, dim=1)
        all_test_pred = torch.cat(all_test_pred, dim=1)

        train_data = MyDataLoader(MyDataset(all_train_pred,
                                            torch.tensor([i[1] for i in self.train_features])),
                                  batch_size=self.args['batch_size'])
        test_data = MyDataLoader(MyDataset(all_test_pred,
                                           torch.tensor([i[1] for i in self.test_features])),
                                 batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        train_data = list(train_data)
        test_data = list(test_data)

        # 将该次的数据存储在本地，可以用来单独测试
        save_graph_pickle((train_data, test_data),
                          os.path.join(self.get_save_path(), 'ensemble_data.pickle'))
        self.start_train(train_data, test_data)

    def load_train_and_test_features_by_diff_files(self, model_name):
        """
        从不同文件中分别加载训练数据以及测试数据
        """
        from processor.ReCAM import load_and_cache_examples, load_and_cache_examples_new
        print(model_name)
        with_gnn = 'GNN' in model_name or 'GAT' in model_name
        if with_gnn:
            self.args['with-kegat'] = True
        print('with_gnn: ', 'Yes' if with_gnn else 'No')
        modelx_name = 'Electra-large'
        if with_gnn:
            modelx_name = 'electra-kegat'
        elif 'Roberta' in model_name:
            modelx_name = 'roberta-large'
        elif model_name == 'Electra':
            modelx_name = 'Electra-large'
        elif model_name == 'Albert':
            modelx_name = 'albert-xxlarge'

        if self.args['subtask_id'] == '1':
            # 如果当前任务是 subtask 1
            train_features = load_and_cache_examples(self.args, modelx_name, self.tokenizer, set_type='train')
            test_features = load_and_cache_examples(self.args, modelx_name, self.tokenizer, set_type='dev')
        elif self.args['subtask_id'] == '2':
            train_features = load_and_cache_examples(self.args, modelx_name, self.tokenizer, set_type='train')
            test_features = load_and_cache_examples(self.args, modelx_name, self.tokenizer, set_type='dev')
        else:
            train_features, test_features = [], []

        train_features_len = len(train_features)
        test_features_len = len(test_features)
        print(train_features_len)
        # data_random_perm 是数据的排列，加载数据一个需 shuffle 一下
        if self.data_random_perm is None:
            randperm_train = torch.randperm(train_features_len)
            randperm_test = torch.randperm(test_features_len)
            self.data_random_perm = (randperm_train, randperm_test)

            # 用固定排列代替，便于 ensemble
            # print('use data random perm: ./logs/data_random_perm_taskA.pickle')
            # self.data_random_perm = load_graph_pickle('./logs/data_random_perm_taskA.pickle')
            # print('use data random perm: ./logs/data_random_perm_taskB.pickle')
            # self.data_random_perm = load_graph_pickle('./logs/data_random_perm_taskB.pickle')

            if os.path.isfile(os.path.join(self.get_save_path(),
                                           'data_random_perm.pickle')):
                # 如果本地已有当前数据的排列，则直接加载
                print('data_random_perm.pickle exist, load it...')
                self.data_random_perm = load_graph_pickle(os.path.join(self.get_save_path(),
                                                                       'data_random_perm.pickle'))
            else:
                # 否则保存此次数据的排列顺序
                print('save data random perm...')
                save_graph_pickle(self.data_random_perm,
                                  os.path.join(self.get_save_path(),
                                               'data_random_perm.pickle'))

        train_features = [train_features[i] for i in self.data_random_perm[0]]
        test_features = [test_features[i] for i in self.data_random_perm[1]]

        self.train_features = train_features
        self.test_features = test_features

    def load_train_and_test_features(self):
        """
        加载训练以及测试数据集，并将其保存进 self.train_features, self.test_features 中
        这里 self.data_random_perm 控制着数据的打乱顺序，且只在第一次创建，因为多个模型面对的应该是相同排序的数据
        """
        from processor.ReCAM import load_and_cache_examples
        modelx_name = 'Electra-large'
        semantic_features = load_and_cache_examples(self.args, modelx_name, self.tokenizer, set_type='train')

        features_len = len(semantic_features)
        # data_random_perm 是数据的排列，加载数据一个需 shuffle 一下
        if self.data_random_perm is None:
            self.data_random_perm = torch.randperm(features_len)

            # 保存此次数据的排列顺序
            save_graph_pickle(self.data_random_perm,
                              os.path.join(self.get_save_path(),
                                           'data_random_perm.pickle'))
        semantic_features = [semantic_features[i] for i in self.data_random_perm]

        self.train_features = semantic_features[:int(features_len * self.args['split_rate'])]
        self.test_features = semantic_features[int(features_len * self.args['split_rate']):]

    def init_pre_exec(self, model_name):
        model_path = self.args['pretrained_model_path']
        if model_name == 'Bert':
            model_path = model_path + 'bert-large-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        elif model_name == 'Electra':
            model_path = model_path + 'Electra-large'
            self.tokenizer = ElectraTokenizer.from_pretrained(model_path)
        elif 'Roberta' in model_name:
            model_path = model_path + 'roberta-large'
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        elif model_name == 'Albert':
            model_path = model_path + 'albert-xxlarge'
            self.tokenizer = AlbertTokenizer.from_pretrained(model_path)
        elif model_name == 'Electra_GNN':
            model_path = model_path + 'Electra-large'
            self.tokenizer = ElectraTokenizer.from_pretrained(model_path)
        else:
            model_path = model_path + 'Electra-large'
            self.tokenizer = ElectraTokenizer.from_pretrained(model_path)
        # 加载训练以及测试数据集
        # self.load_train_and_test_features()
        self.load_train_and_test_features_by_diff_files(model_name)

    def init_model(self, model_name):
        model_path = self.args['pretrained_model_path']
        if model_name == 'Bert':
            model_path = model_path + 'bert-large-uncased'
            self.model = BertForMultipleChoice.from_pretrained(model_path)
        elif model_name == 'Electra' or model_name == 'Electra_trail' or model_name == 'Electra_RELU':
            model_path = model_path + 'Electra-large'
            self.model = ElectraForMultipleChoice.from_pretrained(model_path)
        elif model_name == 'Electra91.3' or model_name == 'Electra_hard':
            model_path = model_path + 'Electra-large'
            self.model = ElectraForMultipleChoiceHard.from_pretrained(model_path)
        elif model_name == 'Electra_Pro':
            model_path = model_path + 'Electra-large'
            self.model = ElectraForMultipleChoicePro.from_pretrained(model_path, is_add_projection=True)
        elif 'Roberta' in model_name:
            model_path = model_path + 'roberta-large'
            self.model = RobertaForMultipleChoice.from_pretrained(model_path)
        elif model_name == 'Albert':
            model_path = model_path + 'albert-xxlarge'
            self.model = AlbertForMultipleChoice.from_pretrained(model_path)
        elif 'GNN' in model_name or 'GAT' in model_name:
            self.model = GAT_goal_model(self.args, is_add_projection=self.args['with_projection_gat'])
        elif 'Electra' in model_name:
            model_path = model_path + 'Electra-large'
            self.model = ElectraForMultipleChoice.from_pretrained(model_path)
        else:
            pass
        self.model.to(self.args['device'])
        if torch.cuda.device_count() > 1 and self.args['use_multi_gpu']:
            print("{} GPUs are available. Let's use them.".format(
                torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

    def single_model_execution(self, model_name):
        """
        执行一个模型的任务，这里先用 model_name if else 做
        最后整理的时候重构
        :param model:
        :return:
        """
        self.args['model_init'] = False
        self.init_pre_exec(model_name)
        create_datasets = Trainer.create_datasets_with_k
        if os.path.isfile(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(model_name))):
            # 如果本地已有当前模型的结果，则直接返回
            print('{} final pred exist, return now...'.format(model_name))
            return load_graph_pickle(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(model_name)))

        k_fold_data = self.get_k_fold_data(self.args['k_fold'], create_datasets)

        # self.test_features，最终预测的数据
        final_test_loader = MyDataLoader(create_datasets(
            self.test_features, shuffle=False),
            batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        final_test_loader = list(final_test_loader)

        final_train_pred = []
        final_test_pred = 0  # 单独切分一部分的预测结果
        for idx, (train_loader, test_loader) in enumerate(k_fold_data):
            print('fold {}/{} start'.format(idx + 1, self.args['k_fold']))
            # 将数据全部放进内存，提升 IO 效率
            train_loader, test_loader = list(train_loader), list(test_loader)
            self.init_model(model_name)

            if os.path.isfile(
                    os.path.join(self.get_save_path(),
                                 '{}_fold_{}.pth'.format(model_name,
                                                         idx))
            ):
                print('{} exist, load state and test...'.format('{}_fold_{}.pth'.format(model_name,
                                                                                        idx)))
                # 如果已存在该模型，则直接加载权重进行测试
                self.model.load_state_dict(torch.load(
                    os.path.join(self.get_save_path(),
                                 '{}_fold_{}.pth'.format(model_name,
                                                         idx))
                ))
                test_loss_opt, acc, test_pred_opt = Trainer.test(self.model,
                                                                 test_loader,
                                                                 self.args)
                # acc, (train_pred_opt, test_pred_opt) = Trainer.train_and_finetune(self.model,
                #                                                                   "5_fold",
                #                                                                   train_loader,
                #                                                                   train_loader,
                #                                                                   test_loader,
                #                                                                   self.args,
                #                                                                   subtask=int(self.args['subtask_id']))
            else:
                print('start train and finetune...')
                # 这里是对五折得到的预测结果存储下来
                acc, (train_pred_opt, test_pred_opt) = Trainer.train_and_finetune(self.model,
                                                                                  "5_fold",
                                                                                  train_loader,
                                                                                  train_loader,
                                                                                  test_loader,
                                                                                  self.args,
                                                                                  subtask=int(self.args['subtask_id']))
                # 保存模型权重
                torch.save(self.model.state_dict(),
                           os.path.join(self.get_save_path(),
                                        '{}_fold_{}.pth'.format(model_name,
                                                                idx)))
            final_train_pred.append(test_pred_opt)

            # 下面是存储单独切分那部分
            test_loss, test_acc, test_pred = Trainer.test(self.model, final_test_loader, self.args)
            final_test_pred = final_test_pred + test_pred
            print('fold: {}/{}, fold acc: {:.4f}, final test acc: {:.4f}'.format(idx + 1,
                                                                                 self.args['k_fold'],
                                                                                 acc,
                                                                                 test_acc))
        final_train_pred = torch.cat(final_train_pred, dim=0)  # 由于是5折，所以把所有数据量完整
        final_test_pred = final_test_pred / len(k_fold_data)
        save_graph_pickle((final_train_pred, final_test_pred),
                          os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(model_name)))
        return final_train_pred, final_test_pred

    def single_model_execution_Non_fold(self, model_name):  # 单折
        """
        执行一个模型的任务，这里先用 model_name if else 做
        最后整理的时候重构
        :param model:
        :return:
        """
        self.args['model_init'] = False
        self.init_pre_exec(model_name)
        create_datasets = Trainer.create_datasets_with_k
        if os.path.isfile(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(model_name))):
            # 如果本地已有当前模型的结果，则直接返回
            print('{} final pred exist, return now...'.format(model_name))
            return load_graph_pickle(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(model_name)))

        # self.test_features，最终预测的数据
        final_test_loader = Trainer.create_datasets(self.test_features,
                                                    shuffle=False,
                                                    batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        final_test_loader = list(final_test_loader)

        train_loader = Trainer.create_datasets(self.train_features,
                                               shuffle=False,
                                               batch_size=self.args['batch_size'])
        train_loader = list(train_loader)

        self.init_model(model_name)

        if os.path.isfile(
                os.path.join(self.get_save_path(),
                             '{}_Non_fold.pth'.format(model_name))
        ):
            print('{} exist, load state and test...'.format('{}_Non_fold.pth'.format(model_name)))
            # 如果已存在该模型，则直接加载权重进行测试
            self.model.load_state_dict(torch.load(
                os.path.join(self.get_save_path(),
                             '{}_Non_fold.pth'.format(model_name))
            ))
            test_loss_opt, acc, test_pred_opt = Trainer.test(self.model,
                                                             train_loader,
                                                             self.args)
        else:
            print('start train and finetune...')
            acc, (train_pred_opt, test_pred_opt) = Trainer.train_and_finetune(self.model,
                                                                              "Non_fold",
                                                                              train_loader,
                                                                              train_loader,
                                                                              train_loader,
                                                                              self.args,
                                                                              subtask=int(self.args['subtask_id']))
            # 保存模型权重
            torch.save(self.model.state_dict(),
                       os.path.join(self.get_save_path(),
                                    '{}_Non_fold.pth'.format(model_name)))
        final_train_pred = test_pred_opt

        # 下面是存储单独切分那部分
        test_loss, test_acc, test_pred = Trainer.test(self.model, final_test_loader, self.args)
        print('Non_fold train acc: {:.4f}, final test acc: {:.4f}'.format(acc, test_acc))

        final_test_pred = test_pred
        save_graph_pickle((final_train_pred, final_test_pred),
                          os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(model_name)))
        return final_train_pred, final_test_pred

    def single_model_execution_diff(self):
        """
        执行一个模型的任务，这里先用 model_name if else 做
        最后整理的时候重构
        :param model:
        :return:
        """

        final_train_pred, final_test_pred = 0, 0

        for md in self.all_model_name:
            if md == "Electra_GAT_RELU":
                self.args['with_projection_gat'] = True
            self.args['model_init'] = False
            self.init_pre_exec(md)
            self.init_model(md)

            if os.path.isfile(
                    os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format('_'.join(self.all_model_name)))):
                # 如果本地已有当前模型的结果，则直接返回
                print('{} final pred exist, return now...'.format('_'.join(self.all_model_name)))
                return load_graph_pickle(
                    os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format('_'.join(self.all_model_name))))

            # self.test_features，最终预测的数据
            final_test_loader = Trainer.create_datasets(self.test_features,
                                                        shuffle=False,
                                                        batch_size=self.args['test_batch_size'])
            # 将数据全部放进内存，提升 IO 效率
            final_test_loader = list(final_test_loader)

            train_loader = Trainer.create_datasets(self.train_features,
                                                   shuffle=False,
                                                   batch_size=self.args['test_batch_size'])
            train_loader = list(train_loader)

            if os.path.isfile(
                    os.path.join(self.get_save_path(),
                                 '{}_Non_fold.pth'.format(md))
            ):
                print('{} exist, load state and test...'.format('{}_Non_fold.pth'.format(md)))
                # 如果已存在该模型，则直接加载权重进行测试
                self.model.load_state_dict(torch.load(
                    os.path.join(self.get_save_path(),
                                 '{}_Non_fold.pth'.format(md))
                ))
                test_loss_opt, acc, test_pred_opt = Trainer.test(self.model,
                                                                 train_loader,
                                                                 self.args)
                # acc = 0
                # test_pred_opt = 0
            else:
                print('start train and finetune...')
                acc, (train_pred_opt, test_pred_opt) = Trainer.train_and_finetune(self.model,
                                                                                  "Non_fold",
                                                                                  train_loader,
                                                                                  train_loader,
                                                                                  train_loader,
                                                                                  self.args,
                                                                                  subtask=int(self.args['subtask_id']))
                # 保存模型权重
                torch.save(self.model.state_dict(),
                           os.path.join(self.get_save_path(),
                                        '{}_Non_fold.pth'.format(md)))
            final_train_pred = final_train_pred + test_pred_opt

            test_loss, test_acc, test_pred = Trainer.test(self.model, final_test_loader, self.args)
            print('Non_fold train acc: {:.4f}, final test acc: {:.4f}'.format(acc, test_acc))
            final_test_pred = final_test_pred + test_pred

        final_train_pred = final_train_pred / len(self.all_model_name)
        # 下面是存储单独切分那部分
        final_test_pred = final_test_pred / len(self.all_model_name)

        save_graph_pickle((final_train_pred, final_test_pred),
                          os.path.join(self.get_save_path(),
                                       '{}_final_pred.pickle'.format('_'.join(self.all_model_name))))
        return final_train_pred, final_test_pred

    def get_k_fold_data(self, k, create_dataset_function):
        """
        获取 k 组数据，其组织格式为 [(train_data1, test_data1), ]
        :param create_dataset_function:
        :param k:
        :return:
        """
        res = []
        split_features_list = self.__k_fold_split(k)
        for i in range(k):
            train_data = []
            for j in split_features_list[:i] + split_features_list[i + 1:]:
                train_data.extend(j)
            test_data = split_features_list[i]

            # 因为一开始 shuffle 过了，这里先设置为 False，以便于调试
            train_loader = MyDataLoader(create_dataset_function(train_data, shuffle=False),
                                        batch_size=self.args['batch_size'])
            test_loader = MyDataLoader(create_dataset_function(test_data, shuffle=False),
                                       batch_size=self.args['test_batch_size'])
            res.append((train_loader, test_loader))
        return res

    def __k_fold_split(self, k):
        """
        将 train_features 切分为 k 块并返回
        :param k:
        :return:
        """
        data_size = len(self.train_features)

        split_features_list = []
        for i in range(k):
            # 切分每块存在 split_features_list 中
            left_idx = i * (data_size // k)
            right_idx = (i + 1) * (data_size // k)
            if i == k - 1:
                right_idx = data_size

            split_batch = self.train_features[left_idx:right_idx]
            split_features_list.append(split_batch)
        return split_features_list

    def exec_evaluation(self):
        """
        使用本地训练好的模型对目标 test data 进行测试
        :return:
        """

        def get_model_and_tokenizer(cls, model_name):
            model = tokenizer = None
            model_path = self.args['pretrained_model_path']
            if model_name == 'Bert':
                model_path = model_path + 'bert-base-uncased'
                model = BertForMultipleChoice.from_pretrained(model_path)
                tokenizer = BertTokenizer.from_pretrained(model_path)
            elif model_name == 'Electra' or model_name == 'Electra1':
                model_path = model_path + 'Electra-large'
                model = ElectraForMultipleChoice.from_pretrained(model_path,
                                                                 is_add_projection=cls.args['with_projection_plm'])
                tokenizer = ElectraTokenizer.from_pretrained(model_path)
            elif model_name == 'Electra91.3' or model_name == 'Electra_hard':
                model_path = model_path + 'Electra-large'
                model = ElectraForMultipleChoiceHard.from_pretrained(model_path,
                                                                     is_add_projection=cls.args['with_projection_plm'])
                tokenizer = ElectraTokenizer.from_pretrained(model_path)
            elif 'Roberta' in model_name:
                model_path = model_path + 'roberta-large'
                model = RobertaForMultipleChoice.from_pretrained(model_path)
                tokenizer = RobertaTokenizer.from_pretrained(model_path)
            elif model_name == 'Albert':
                model_path = model_path + 'albert-xxlarge'
                model = AlbertForMultipleChoice.from_pretrained(model_path)
                tokenizer = AlbertTokenizer.from_pretrained(model_path)
            elif 'GNN' in model_name or 'GAT' in model_name:
                model_path = model_path + 'Electra-large'
                tokenizer = ElectraTokenizer.from_pretrained(model_path)
                model = GAT_goal_model(cls.args, is_add_projection=cls.args['with_projection_gat'])
            else:
                pass
            return model, tokenizer

        from processor.ReCAM import load_and_cache_examples
        create_datasets = Trainer.create_datasets_with_k
        get_all_features_from_task_x = None
        if self.args['subtask_id'] == '1':
            predict_csv_path = 'datasets/Task1/Task1_1_test.jsonl'
        elif self.args['subtask_id'] == '2':
            predict_csv_path = 'datasets/Task2/Task1_2_test.jsonl'
        else:
            predict_csv_path = 'datasets/Task1/Task1_1_test.jsonl'

        all_test_pred = []
        for model_name in self.all_model_name:
            # 根据 model_name 加载相应的 model 以及 tokenizer
            model, tokenizer = get_model_and_tokenizer(self, model_name)
            # 从本地加载最终需要测试的数据
            print(model_name)
            with_gnn = 'GNN' in model_name or 'GAT' in model_name
            if with_gnn:
                self.args['with-kegat'] = True
            print('with_gnn: ', 'Yes' if with_gnn else 'No')
            modelx_name = 'Electra-large'
            if with_gnn:
                modelx_name = 'electra-kegat'
            elif 'Roberta' in model_name:
                modelx_name = 'roberta-large'
            elif model_name == 'Electra':
                modelx_name = 'Electra-large'
            elif model_name == 'Albert':
                modelx_name = 'albert-xxlarge'

            test_features = load_and_cache_examples(self.args, modelx_name, tokenizer, set_type='test')
            # 利用 test_features 创建 DataLoader，shuffle 设置为 False
            final_test_loader = MyDataLoader(create_datasets(
                test_features, shuffle=False),
                batch_size=self.args['test_batch_size'])
            # 将数据全部放进内存，提升 IO 效率
            final_test_loader = list(final_test_loader)

            final_test_pred = 0
            for fold_idx in range(self.args['k_fold']):
                print('predict fold {}/{}'.format(fold_idx + 1, self.args['k_fold']))
                # 遍历加载每一折的模型进行预测
                model_path = os.path.join(self.get_save_path(), '{}_fold_{}.pth'.format(model_name, fold_idx))
                model.load_state_dict(torch.load(model_path))
                model.to(self.args['device'])

                test_loss, test_acc, test_pred = Trainer.test(model, final_test_loader, self.args)
                final_test_pred = final_test_pred + test_pred

            final_test_pred = final_test_pred / self.args['k_fold']
            save_graph_pickle((final_test_pred, final_test_pred),
                              os.path.join(self.get_save_path(),
                                           '{}_final_pred.pickle'.format('_'.join(self.all_model_name))))
            assert 1 == 2
            all_test_pred.append(final_test_pred)
        # all_test_pred 为多个模型预测结果的横向拼接
        all_test_pred = torch.cat(all_test_pred, dim=1)

        # 接下来是使用 stacking model 进行预测
        print('Start stackingNet model predict...')
        test_data = MyDataLoader(MyDataset(all_test_pred,
                                           torch.zeros(all_test_pred.size(0), dtype=torch.int64)),
                                 batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        test_data = list(test_data)

        stacking_model = StackingNNet(test_data[0][0].size(1), test_data[0][0].size(1) // len(self.all_model_name))
        stacking_model.load_state_dict(torch.load(
            os.path.join(self.get_save_path(),
                         'stacking_model.pth')
        ))
        stacking_model.to(self.args['device'])
        _, _, test_pred = self.test(stacking_model, test_data)
        # ans 为最终预测的结果
        ans = test_pred.softmax(dim=1).argmax(dim=1).cpu().numpy()
        id_list = np.arange(len(ans))
        globals()['ans'] = ans
        data_all = pd.DataFrame(np.stack((id_list, ans)).T, columns=['id', 'label'])
        data_all.to_csv(
            os.path.join(self.get_save_path(), 'subtask1.csv'),
            columns=['id', 'label'],
            header=False,
            index=False
        )
        return ans

    def exec_evaluation_new(self):
        all_test_pred = []
        print('data_random_perm.pickle exist, load it...')
        self.data_random_perm = load_graph_pickle(os.path.join(self.get_save_path(),
                                                               'data_random_perm.pickle'))
        xb = torch.argsort(self.data_random_perm[1])  # 还原下标
        for i in self.all_model_name:
            self.init_pre_exec(i)
            # 将各种不同模型的结果整合起来
            final_pred = load_graph_pickle(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(i)))
            recover_pred = final_pred[1][xb]
            all_test_pred.append(recover_pred)

        all_test_pred = torch.cat(all_test_pred, dim=1)
        print('Start stackingNet model predict...')
        # test_data = MyDataLoader(MyDataset(all_test_pred,
        #                                    torch.zeros(all_test_pred.size(i), dtype=torch.int64)),
        #                          batch_size=self.args['test_batch_size'])

        label = torch.tensor([i[1] for i in self.test_features])
        test_data = MyDataLoader(MyDataset(all_test_pred, label[xb]), batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        test_data = list(test_data)
        print(len(test_data))
        print(test_data[0][0].size(1))
        stacking_model = StackingNNet(test_data[0][0].size(1), test_data[0][0].size(1) // len(self.all_model_name))
        stacking_model.load_state_dict(torch.load(
            os.path.join(self.get_save_path(),
                         'stacking_model.pth')
        ))
        stacking_model.to(self.args['device'])
        _, test_acc, test_pred = self.test(stacking_model, test_data)
        # ans 为最终预测的结果
        ans = test_pred.softmax(dim=1).argmax(dim=1).cpu().numpy()

        id_list = np.arange(len(ans))
        print(ans)
        print("test_acc: ", test_acc)
        globals()['ans'] = ans
        data_all = pd.DataFrame(np.stack((id_list, ans)).T, columns=['id', 'label'])
        data_all.to_csv(
            os.path.join(self.get_save_path(), 'subtask2.csv'),
            columns=['id', 'label'],
            header=False,
            index=False
        )
        return ans

    def exec_evaluation_merge(self):
        all_test_pred = []
        print('data_random_perm.pickle exist, load it...')
        self.data_random_perm = load_graph_pickle(os.path.join(self.get_save_path(),
                                                               'data_random_perm.pickle'))
        xb = torch.argsort(self.data_random_perm[1])  # 还原下标
        for i in self.all_model_name:
            self.init_pre_exec(i)
            # 将各种不同模型的结果整合起来
            final_pred = load_graph_pickle(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(i)))
            recover_pred = final_pred[1][xb]
            all_test_pred.append(recover_pred)

        all_test_pred = torch.cat(all_test_pred, dim=1)
        print('Start stackingNet model predict...')
        # test_data = MyDataLoader(MyDataset(all_test_pred,
        #                                    torch.zeros(all_test_pred.size(i), dtype=torch.int64)),
        #                          batch_size=self.args['test_batch_size'])

        label = torch.tensor([i[1] for i in self.test_features])
        test_data = MyDataLoader(MyDataset(all_test_pred, label[xb]), batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        test_data = list(test_data)
        print(len(test_data))
        print(test_data[0][0].size(1))
        stacking_model = StackingNNet(test_data[0][0].size(1), test_data[0][0].size(1) // len(self.all_model_name))
        stacking_model.load_state_dict(torch.load(
            os.path.join(self.get_save_path(),
                         'stacking92.59_model.pth')
        ))
        stacking_model.to(self.args['device'])
        _, test_acc, test_pred1 = self.test(stacking_model, test_data)

        all_test_pred = []

        for i in self.all_model_name2:
            self.init_pre_exec(i)
            # 将各种不同模型的结果整合起来
            final_pred = load_graph_pickle(os.path.join(self.get_save_path(), '{}_final_pred.pickle'.format(i)))
            recover_pred = final_pred[1][xb]
            all_test_pred.append(recover_pred)

        all_test_pred = torch.cat(all_test_pred, dim=1)
        print('Start stackingNet model predict...')
        # test_data = MyDataLoader(MyDataset(all_test_pred,
        #                                    torch.zeros(all_test_pred.size(i), dtype=torch.int64)),
        #                          batch_size=self.args['test_batch_size'])

        label = torch.tensor([i[1] for i in self.test_features])
        test_data = MyDataLoader(MyDataset(all_test_pred, label[xb]), batch_size=self.args['test_batch_size'])
        # 将数据全部放进内存，提升 IO 效率
        test_data = list(test_data)
        print(len(test_data))
        print(test_data[0][0].size(1))
        stacking_model = StackingNNet(test_data[0][0].size(1), test_data[0][0].size(1) // len(self.all_model_name2))
        stacking_model.load_state_dict(torch.load(
            os.path.join(self.get_save_path(),
                         'stacking92.48_model.pth')
        ))
        stacking_model.to(self.args['device'])
        _, test_acc, test_pred2 = self.test(stacking_model, test_data)
        print(test_pred2.shape)
        test_pred = test_pred1 + test_pred2
        print(test_pred.shape)
        # ans 为最终预测的结果
        ans = test_pred.softmax(dim=1).argmax(dim=1).cpu().numpy()

        id_list = np.arange(len(ans))
        # print(ans)
        real_label_path = r'/mnt/ssd/wwwlps/Task4/SemEval2021/Answer/subtask2_real.csv'
        real_label_list = pd.read_csv(real_label_path, header=None).values[:, 1]
        index = np.arange(len(ans))
        print(len(index))
        real_acc = len(index[ans == real_label_list]) / len(ans)
        print("final_acc: ", real_acc)
        globals()['ans'] = ans
        data_all = pd.DataFrame(np.stack((id_list, ans)).T, columns=['id', 'label'])
        data_all.to_csv(
            os.path.join(self.get_save_path(), 'subtask2.csv'),
            columns=['id', 'label'],
            header=False,
            index=False
        )
        return ans


# %%
if __name__ == '__main__':
    from transformers import ElectraTokenizer
    import os
    import torch
    from config import args, project_root_path

    os.chdir(project_root_path)

    stacking = Stacking(args)
    self = stacking
    # self.args['exec_time'] = '2020.02.23-21.16.56'
    stacking.start()
    stacking.exec_evaluation()
    # stacking.start_with_ensemble_data('logs/Roberta_2020.02.23-18.16.09')

    # net = StackingNNet(10, 3)

    # def load_graph_pickle(fpath):
    #     graph_zip = None
    #     with open(fpath, 'rb') as f:
    #         graph_zip = pickle.load(f)
    #     return graph_zip

    # train_loader, test_loader = load_graph_pickle('ensemble_inputData1.pickle')
    # stacking_model = StackingNNet(train_loader[0][0].shape[1], 3)
    # stacking_model.to(self.args['device'])
    # optimizer = optim.Adam(stacking_model.parameters(), lr=self.args['ensemble_lr'])
    #
    # acc = 0
    # for epoch in range(self.args['ensemble_epochs']):
    #     train_loss, train_acc, train_pred = self.train(stacking_model, train_loader, optimizer)
    #     test_loss, test_acc, test_pred = self.test(stacking_model, test_loader)
    #     acc = max(acc, test_acc)
    # print('final acc: {:.4f}'.format(acc))
    #
    # res = 0
    # count_all = 0
    # for i in range(len(train_loader)):
    #     count_all += len(train_loader[i][1])
    #     res = res + train_loader[i][0][:, 3:].softmax(dim=1).argmax(dim=1).cpu().eq(train_loader[i][1]).sum().item()
    # print(res * 1.0 / count_all * 100)
