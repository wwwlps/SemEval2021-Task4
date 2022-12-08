import torch
import json
import re
import os
from utils.graph_utils import GraphUtils


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    pop_label = True
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(1)
        else:
            tokens_b.pop(1)


def get_features_from_RACE(data_path, tokenizer, max_seq_length, proportion=4, num_choices=4):
    features = []
    batch_text_or_text_pairs = []
    label_list = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        mlen = 0
        cnt = 0
        for i in f.readlines():
            cnt += 1
            if cnt % proportion != 0:  # 数据量比较大，筛选
                continue
            d = json.loads(i)
            article_tokens = d['article']
            question_tokens = d['question']
            placeholder_num = len(re.findall(r"_", question_tokens))
            if placeholder_num > 1:
                continue
            mlen = max(mlen, len(article_tokens.split(' ')))
            choices_list = [d['option_0'], d['option_1'], d['option_2'], d['option_3']]
            label_list.append(d['label'])
            tem_question = question_tokens
            for j in choices_list:
                ending_tokens = j
                ending_tokens = re.sub(r"\\", "", ending_tokens)
                if len(re.findall(r"_", tem_question)) == 0:
                    question_tokens = tem_question + ending_tokens
                else:
                    question_tokens = re.sub(r"_", ending_tokens, tem_question)
                batch_text_or_text_pairs.append(
                    ('Q: ' + article_tokens, 'A: ' + question_tokens))
    token_encode = tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs,
                                               add_special_tokens=True,
                                               max_length=max_seq_length,
                                               padding=True,
                                               truncation="only_first",
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
    input_ids = token_encode.get('input_ids').reshape(-1, num_choices, max_seq_length)
    token_type_ids = token_encode.get('token_type_ids').reshape(-1, num_choices, max_seq_length)
    attention_mask = token_encode.get('attention_mask').reshape(-1, num_choices, max_seq_length)

    for i in range(input_ids.size(0)):
        choices_features = []
        for j in range(num_choices):
            choices_features.append(
                {'tokens': tokenizer.convert_ids_to_tokens(input_ids[i][j]),
                 'input_ids': input_ids[i][j],
                 'attention_mask': attention_mask[i][j],
                 'token_type_ids': token_type_ids[i][j],
                 'position_ids': torch.arange(2, 2 + max_seq_length) if 'Roberta' in str(
                     type(tokenizer)) else torch.arange(
                     max_seq_length),
                 }
            )
        features.append((choices_features, label_list[i]))
    return features


def get_features_from_ReCAM(data_path, tokenizer, max_seq_length, set_type, num_choices=5):
    """
    从 ReCAM 中获取数据，做 choices_num 分类问题所需数据
    :param tokenizer:
    :param max_seq_length:
    :return:
    """
    features = []
    batch_text_or_text_pairs = []
    label_list = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        cnt = 0
        for i in f.readlines():
            d = json.loads(i)
            article_tokens = d['article']
            question_tokens = d['question']
            cnt += 1
            choices_list = [d['option_0'], d['option_1'], d['option_2'], d['option_3'], d['option_4']]
            if set_type == 'test':
                label_list.append(0)
            else:
                label_list.append(d['label'])
            tem_question = question_tokens
            for j in choices_list:
                ending_tokens = j
                insert_tokens = '[SEP] ' + ending_tokens + ' [SEP]'
                # insert_tokens = ending_tokens
                question_tokens = re.sub(r'@placeholder', insert_tokens,
                                         tem_question)  # 将选项插入到问题中的@placeholder中,QA作为一个整体
                batch_text_or_text_pairs.append(('Q: ' + article_tokens, 'A: ' + question_tokens))
    token_encode = tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs,
                                               add_special_tokens=True,
                                               max_length=max_seq_length,
                                               padding=True,
                                               truncation="only_first",
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
    input_ids = token_encode.get('input_ids').reshape(-1, num_choices, max_seq_length)
    token_type_ids = token_encode.get('token_type_ids').reshape(-1, num_choices, max_seq_length)
    attention_mask = token_encode.get('attention_mask').reshape(-1, num_choices, max_seq_length)

    for i in range(input_ids.size(0)):
        choices_features = []
        for j in range(num_choices):
            choices_features.append(
                {'tokens': tokenizer.convert_ids_to_tokens(input_ids[i][j]),
                 'input_ids': input_ids[i][j],
                 'attention_mask': attention_mask[i][j],
                 'token_type_ids': token_type_ids[i][j],
                 'position_ids': torch.arange(2, 2 + max_seq_length) if 'Roberta' in str(
                     type(tokenizer)) else torch.arange(
                     max_seq_length),
                 }
            )
        features.append((choices_features, label_list[i]))
    return features


def get_features_from_ReCAM_Roberta(data_path, tokenizer, max_seq_length, set_type, num_choices=5):
    """
    从 ReCAM 中获取数据，做 choices_num 分类问题所需数据
    :param tokenizer:
    :param max_seq_length:
    :return:
    """
    features = []
    batch_text_or_text_pairs = []
    label_list = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        cnt = 0
        for i in f.readlines():
            d = json.loads(i)
            article_tokens = d['article']
            question_tokens = d['question']
            cnt += 1
            choices_list = [d['option_0'], d['option_1'], d['option_2'], d['option_3'], d['option_4']]
            labelx = 0
            if set_type == 'test':
                label_list.append(0)
                labelx = 0
            else:
                label_list.append(d['label'])
                labelx = d['label']
            tem_question = question_tokens
            article_tokens = tokenizer.tokenize(article_tokens)
            choices_features = []
            for j in choices_list:
                ending_tokens = j
                question_tokens = re.sub(r'@placeholder', ending_tokens,
                                         tem_question)  # 将选项插入到问题中的@placeholder中,QA作为一个整体
                question_tokens = tokenizer.tokenize(question_tokens)
                tt = 'Ġ' + ending_tokens
                if tt in question_tokens:
                    pos = question_tokens.index(tt)
                    question_tokens.insert(pos, tokenizer.sep_token)
                    question_tokens.insert(pos + 2, tokenizer.sep_token)
                _truncate_seq_pair(article_tokens, question_tokens, max_seq_length - 3)

                all_tokens = [tokenizer.cls_token] + article_tokens + [tokenizer.sep_token] + [tokenizer.sep_token] + question_tokens + [tokenizer.sep_token]
                token_type_ids = [0] * (len(article_tokens) + 2) + [0] * (len(question_tokens) + 1)
                input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
                attention_mask = [1] * len(input_ids)

                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += [0] * (max_seq_length - len(input_ids))
                attention_mask += padding
                token_type_ids += padding

                input_ids = torch.tensor(input_ids)
                attention_mask = torch.tensor(attention_mask)
                token_type_ids = torch.tensor(token_type_ids)

                assert len(input_ids) == max_seq_length
                assert len(attention_mask) == max_seq_length
                assert len(token_type_ids) == max_seq_length

                choices_features.append(
                    {'tokens': all_tokens,
                     'input_ids': input_ids,
                     'attention_mask': attention_mask,
                     'token_type_ids': token_type_ids,
                     'position_ids': torch.arange(2, 2 + max_seq_length) if 'Roberta' in str(
                         type(tokenizer)) else torch.arange(
                         max_seq_length),
                     }
                )
            features.append((choices_features, labelx))
    return features


def get_binary_features_from_ReCAM(data_path, tokenizer, max_seq_length, set_type, num_choices=5):
    """
    single choice
    二分类
    """
    features = []
    batch_text_or_text_pairs = []
    label_list = []
    sub_label_list = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        cnt = 0
        for i in f.readlines():
            d = json.loads(i)
            article_tokens = d['article']
            question_tokens = d['question']
            cnt += 1
            choices_list = [d['option_0'], d['option_1'], d['option_2'], d['option_3'], d['option_4']]
            label_list.append(d['label'])
            sub_label = [1 if t == int(d['label']) else 0 for t in range(num_choices)]
            sub_label_list.append(sub_label)
            tem_question = question_tokens
            for j in choices_list:
                ending_tokens = j
                question_tokens = re.sub(r'@placeholder', ending_tokens,
                                         tem_question)  # 将选项插入到问题中的@placeholder中,QA作为一个整体
                batch_text_or_text_pairs.append(('Q: ' + article_tokens, 'A: ' + question_tokens))
    token_encode = tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs,
                                               add_special_tokens=True,
                                               max_length=max_seq_length,
                                               padding=True,
                                               truncation="only_first",
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
    input_ids = token_encode.get('input_ids').reshape(-1, max_seq_length)
    token_type_ids = token_encode.get('token_type_ids').reshape(-1, max_seq_length)
    attention_mask = token_encode.get('attention_mask').reshape(-1, max_seq_length)

    sub_label_list = [i for j in sub_label_list for i in j]
    assert len(sub_label_list) == input_ids.size(0)
    for i in range(input_ids.size(0)):
        choices_features = [{'tokens': tokenizer.convert_ids_to_tokens(input_ids[i]),
                             'input_ids': input_ids[i],
                             'attention_mask': attention_mask[i],
                             'token_type_ids': token_type_ids[i],
                             'position_ids': torch.arange(2, 2 + max_seq_length) if 'Roberta' in str(
                                 type(tokenizer)) else torch.arange(
                                 max_seq_length),
                             }]
        features.append((choices_features, sub_label_list[i]))
    return features


def get_features_from_ReCAM_with_kbert(data_path, tokenizer, max_seq_length, set_type, num_choices=5):
    from processor.global_processor import add_knowledge_with_vm
    graph = GraphUtils()
    print('graph init...')
    graph.load_mp_all_by_pickle(graph.args['mp_pickle_path'])
    # graph.init(is_load_necessary_data=True)   # 只需要边的关系，因此不需要加载权重
    print('merge graph by downgrade...')
    graph.merge_graph_by_downgrade()
    print('reduce graph noise...')
    graph.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
    print('reduce graph noise done!')

    features = []
    with open(data_path, 'r') as f:
        for i in f.readlines():
            d = json.loads(i)
            article_tokens = d['article']
            question_tokens = d['question']
            choices_list = [d['option_0'], d['option_1'], d['option_2'], d['option_3'], d['option_4']]
            label = d['label']
            choices_features = []
            tem_question = question_tokens
            for j in choices_list:
                ending_tokens = j
                question_tokens = re.sub(r'@placeholder', ending_tokens, tem_question)
                article_len = max_seq_length - len(tokenizer.tokenize(question_tokens)) * 2 - 20  # 截取文章的长度
                # article_len = (max_seq_length - len(tokenizer.tokenize(question_tokens))) // 2 - 40  # 截取文章的长度
                select_article = article_tokens[:article_len]
                source_sent = '{} {} {} {} {}'.format(tokenizer.cls_token,
                                                      select_article,
                                                      tokenizer.sep_token,
                                                      question_tokens,
                                                      tokenizer.sep_token,
                                                      )
                tokens, soft_pos_id, attention_mask, segment_ids = add_knowledge_with_vm(
                    mp_all=graph.mp_all,
                    sent_batch=[source_sent],
                    tokenizer=tokenizer,
                    article=select_article,
                    question=question_tokens,
                    option=ending_tokens,
                    max_entities=2,
                    max_length=max_seq_length)
                tokens = tokens[0]
                soft_pos_id = torch.tensor(soft_pos_id[0])
                attention_mask = torch.tensor(attention_mask[0])
                segment_ids = torch.tensor(segment_ids[0])
                input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

                assert input_ids.shape[0] == max_seq_length
                assert attention_mask.shape[0] == max_seq_length
                assert soft_pos_id.shape[0] == max_seq_length
                assert segment_ids.shape[0] == max_seq_length

                if 'Roberta' in str(type(tokenizer)):
                    # 这里做特判是因为 Roberta 的 Embedding pos_id 是从 2 开始的
                    # 而 Bert 是从零开始的
                    soft_pos_id = soft_pos_id + 2

                choices_features.append(
                    {'tokens': tokens,
                     'input_ids': input_ids,
                     'attention_mask': attention_mask,
                     'token_type_ids': segment_ids,
                     'position_ids': soft_pos_id,
                     }
                )
            features.append((choices_features, label))
    return features


def get_graph_features_from_task_solo(data_path, set_type):
    import torch_geometric.data
    graph = GraphUtils()
    print('graph init...')
    graph.init(is_load_necessary_data=True)
    print('merge graph by downgrade...')
    graph.merge_graph_by_downgrade()
    print('reduce graph noise...')
    graph.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
    print('reduce graph noise done!')

    with open(data_path, 'r', encoding='UTF-8') as f:
        cnt = 0
        features = []
        for i in f.readlines():
            d = json.loads(i)
            article_tokens = d['article']
            question_tokens = d['question']
            cnt += 1
            choices_list = [d['option_0'], d['option_1'], d['option_2'], d['option_3'], d['option_4']]
            if set_type == 'test':
                label = 0
            else:
                label = d['label']
            context_tokens = question_tokens
            for j in range(5):
                ending_tokens = choices_list[j]
                mp = graph.get_submp_by_sentences([context_tokens, ending_tokens], is_merge=True)[0]
                '''
                x: 与 context_tokens, ending_tokens 相关的节点的表示
                x_index: context_tokens, ending_tokens 里存在的节点 idx
                edge_index: 边信息
                edge_weight: 边权重
                '''
                x, x_index, edge_index, edge_weight = graph.encode_index(mp)
                if x.size(0) == 0:  # 当匹配不到实体时，弥补一个空的
                    x = torch.tensor([graph.get_default_oov_feature()])
                    x_index = torch.ones(len(x), dtype=torch.bool)
                data = torch_geometric.data.Data(x=x, pos=x_index, edge_index=edge_index, edge_attr=edge_weight,
                                                 y=torch.tensor([int(label == j)]))
                features.append(data)
    return features


def get_graph_features_from_task(data_path, set_type):
    from torch_geometric.data import DataLoader
    features = get_graph_features_from_task_solo(data_path, set_type)
    features = list(DataLoader(features, batch_size=5, shuffle=False))
    return features


def get_features_from_task(data_path, tokenizer, max_seq_length, set_type, with_k_bert=False, with_gnn=False):
    get_semantic_function = get_features_from_ReCAM_with_kbert if with_k_bert else get_features_from_ReCAM
    semantic_features = get_semantic_function(data_path=data_path,
                                              tokenizer=tokenizer,
                                              max_seq_length=max_seq_length,
                                              set_type=set_type)
    x = [i[0] for i in semantic_features]  # semantic_features
    if with_gnn:
        graph_features = get_graph_features_from_task(data_path, set_type)
        x = list(zip(x, graph_features))  # 组合两个 features
    y = [i[1] for i in semantic_features]  # 分离标签
    return [(x[i], y[i]) for i in range(len(y))]


def load_and_cache_examples(args, model_name, tokenizer, set_type='train'):
    subtask_id = int(args['subtask_id'])
    cached_features_file = os.path.join(args['data_dir'][subtask_id - 1], 'pre_weights/cached_{}_{}_{}_{}'.format(
        set_type,
        model_name,
        str(args['max_seq_length']),
        str(subtask_id)))
    data_path = os.path.join(args['data_dir'][subtask_id - 1], 'Task_{}_{}.jsonl'.format(
        str(subtask_id),
        set_type))
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        features = get_features_from_task(data_path,
                                          tokenizer,
                                          args['max_seq_length'],
                                          set_type,
                                          with_k_bert=args['with_kemb'],
                                          with_gnn=args['with_kegat'], )
        torch.save(features, cached_features_file)
    return features


def load_and_cache_examples_binary(args, model_name, tokenizer, set_type='train'):
    subtask_id = int(args['subtask_id'])

    data_path = os.path.join(args['data_dir'][subtask_id - 1], 'Task_{}_{}.jsonl'.format(
        str(subtask_id),
        set_type))

    features = get_binary_features_from_ReCAM(data_path,
                                              tokenizer,
                                              args['max_seq_length'], )

    return features


def load_and_cache_examples_Roberta(args, model_name, tokenizer, set_type='train'):
    subtask_id = int(args['subtask_id'])
    cached_features_file = os.path.join(args['data_dir'][subtask_id - 1], 'pre_weights/cached_{}_{}_{}_{}'.format(
        set_type,
        model_name,
        str(args['max_seq_length']),
        str(subtask_id)))
    data_path = os.path.join(args['data_dir'][subtask_id - 1], 'Task_{}_{}.jsonl'.format(
        str(subtask_id),
        set_type))
    # if os.path.exists(cached_features_file):
    #     features = torch.load(cached_features_file)
    # else:
    features = get_features_from_ReCAM_Roberta(data_path,
                                               tokenizer,
                                               args['max_seq_length'],
                                               set_type, )
        # torch.save(features, cached_features_file)
    return features


def load_and_cache_examples_new(args, model_name, tokenizer, set_type='train'):
    subtask_id = int(args['subtask_id'])
    cached_features_file = os.path.join(args['data_dir'][subtask_id - 1], 'pre_weights/cachedTrail_{}_{}_{}_{}'.format(
        set_type,
        model_name,
        str(args['max_seq_length']),
        str(subtask_id)))
    data_path = os.path.join(args['data_dir'][subtask_id - 1], 'Task_{}_addtrail_{}.jsonl'.format(
        str(subtask_id),
        set_type))
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        features = get_features_from_task(data_path,
                                          tokenizer,
                                          args['max_seq_length'],
                                          set_type,
                                          with_k_bert=args['with_kemb'],
                                          with_gnn=args['with_kegat'], )
        torch.save(features, cached_features_file)
    return features


def load_and_cache_examples_from_RACE(args, model_name, tokenizer, set_type='train', level='middle', proportion=6):
    cached_features_file = os.path.join('datasets/', 'pre_weights_RACE/cached_{}_{}_{}_{}'.format(
        set_type,
        model_name,
        str(args['max_seq_length']),
        str(proportion)))
    data_path = os.path.join('datasets/RACE/', '{}/{}.jsonl'.format(set_type, level))
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        features = get_features_from_RACE(data_path,
                                          tokenizer,
                                          args['max_seq_length'],
                                          proportion=proportion)
        torch.save(features, cached_features_file)
    return features


if __name__ == '__main__':
    import os
    import torch

    os.chdir('..')

    # from transformers import RobertaTokenizer, ElectraTokenizer
    #
    # a = [1, 2, 3]
    # print(a + [RobertaTokenizer.sep_token])
    #
    model_path = r"D:/transformer_files/Roberta-large"
    # tokenizer = ElectraTokenizer.from_pretrained(model_path)
    # train_data = get_features_from_ReCAM('datasets/Task2/Task_2_dev.jsonl',
    #                                      tokenizer=tokenizer,
    #                                      max_seq_length=160,
    #                                      set_type='dev')
    # x, y = train_data[0]
    # print(x)
    # print(y)

    # input_ids = torch.stack([i.get('input_ids') for i in x], dim=0).reshape(
    #     -1, 5, x[0].get('input_ids').size(0))
    # attention_mask = torch.stack([i.get('attention_mask') for i in x], dim=0).reshape(
    #     -1, 5, x[0].get('attention_mask').size(0))
    # token_type_ids = torch.stack([i.get('token_type_ids') for i in x], dim=0).reshape(
    #     -1, 5, x[0].get('token_type_ids').size(0))
    # print(token_type_ids.shape)

    from transformers import RobertaTokenizer, RobertaForMultipleChoice
    import torch

    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForMultipleChoice.from_pretrained(model_path)
    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    choice0 = "It is eaten with a fork and a knife."
    choice1 = "It is eaten while held in the hand."
    labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
    encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True,  return_token_type_ids=True, )
    print(encoding.get('token_type_ids'))