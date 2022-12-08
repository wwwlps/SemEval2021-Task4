import os
import argparse
from config import args as default_args  # 导入config里的默认参数
from config import project_root_path, Baseline_model
from utils.trainer import Trainer
from processor.ReCAM import *
from src.ensemble import Stacking
from transformers import (
    RobertaTokenizer,
    AlbertConfig,
    ElectraTokenizer,
    BartConfig,
)
from model.baseline import ElectraForMultipleChoice, AlbertForMultipleChoice, \
    RobertaForMultipleChoice, BertForMultipleChoice
from model.GAT_goal import GAT_goal_model
from model.multi_task import ElectraForMultipleChoice_Mul
from model.DCMN_goal import ElectraForMultipleChoiceWithMatch
from processor.ReCAM_DCMN import load_and_cache_examples_DCMN

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
    parser.add_argument('--subtask-id', type=str, default=default_args['subtask_id'],
                        required=False, choices=['1', '2'],
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

    args['exec_time'] = "2021.01.08-10.53.59"

    stacking = Stacking(args)
    self = stacking
    # ensemble_folder = r"../logs1/Electra_Electra_hard_Roberta_2021.01.08-10.53.59"
    # stacking.start()
    # stacking.single_model_execution_Non_fold(self.all_model_name[0])
    # stacking.single_model_execution_diff()
    # stacking.exec_evaluation_new()
    stacking.exec_evaluation_merge()
    # stacking.exec_evaluation()
    # ensemble_folder = "../logs/Electra_Electra_GNN_Electra_GAT_RELU_91.8_2021.01.08-10.53.59"
    # stacking.start_with_final_pred()
    # stacking.start_with_ensemble_data(ensemble_folder)

    # stacking.exec_evaluation()


