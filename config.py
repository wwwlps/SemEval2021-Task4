import torch
import time
from transformers import (
    BertModel,
    BertTokenizer,
    BertConfig,
    AlbertModel,
    AlbertTokenizer,
    AlbertConfig,
    ElectraModel,
    ElectraTokenizer,
    ElectraConfig,
    RobertaModel,
    RobertaTokenizer,
    RobertaConfig,
)

project_root_path = r'/mnt/ssd/wwwlps/Task4/SemEval2021'   # 项目目录
args = {
    'batch_size': 1,  # 使用了梯度累积，实际batch_size等于args['batch_size']*accumulation_steps,
    # 这份代码里是1*16, 所以实际的batch_size效果是16,大家默认使用就行
    'test_batch_size': 4,
    'lr': 0.001,
    'fine_tune_lr': 0.000005,
    'ensemble_lr': 0.001,
    'adam_epsilon': 0.000001,
    'weight_decay': 0.01,
    'warmup_proportion': 0.05,  # warmup 比例，这里只控制了前2个epoch，fine_tune阶段代码里设置的是0.1
    'epochs': 2,
    'fine_tune_epochs': 10,
    'ensemble_epochs': 32,
    'k_fold': 5,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'use_multi_gpu': False,
    'split_rate': 0.8,
    'max_seq_length': 210,
    'accumulation_steps': 16,
    'unfixed_layer': 10,
    'exec_time': time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime()),  # 执行程序的时间，主要用来保存最优 model
    'model_init': False,  # 是否在每次 train_and_finetune 之前对 model 初始化，如果要做五折交叉验证建议设置为 True
    'is_save_checkpoints': False,  # 是否保存权重信息
    'checkpoints_dir': '../checkpoints',  # task2的权重目录
    'checkpoints_dir1': '../checkpoints1',  # task1的权重目录
    'checkpoints_Mnli': '../checkpoints_Mnli',
    'data_dir': ['datasets/Task1/', 'datasets/Task2/'],
    'with_projection_plm': False,
    'with_projection_gat': False,
    'with_kegat': False,
    'with_kemb': False,
    'with_MstageMtask': False,
    'with_DCMN': False,
    'pretrained_model_path': '../transformer_files/',
    'subtask_id': '2',  # 执行哪个子任务 ['1', '2']
    'is_save_logs': False,  # 是否保存 tensorboard logs 信息
    'logs_dir': '../logs/',
    'logs_dir1': '../logs1/',
}
pretrained_model_path = '../transformer_files/'
Baseline_model = {
    'electra': {
        '{}{}'.format(pretrained_model_path, 'Electra-large'),
    },
    'roberta': {
        'model_path': '{}{}'.format(pretrained_model_path, 'roberta-large'),
        'tokenizer': RobertaTokenizer
    },
    'bert': {
        'model_path': '{}{}'.format(pretrained_model_path, 'bert-large-uncased'),
        'tokenizer': BertTokenizer
    },
    'albert': {
        'model_path': '{}{}'.format(pretrained_model_path, 'albert-xxlarge'),
        'tokenizer': AlbertTokenizer
    }
}

