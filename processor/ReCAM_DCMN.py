import torch
import json
import os
import re


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


def DCMN_get_features_from_ReCAM(data_path, tokenizer, max_seq_length, num_choices=5):
    """
    从 ReCAM 中获取数据，做 choices_num 分类问题所需数据
    :param tokenizer:
    :param max_seq_length:
    :return:
    """
    features = []
    label_list = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        cnt = 0
        for i in f.readlines():
            d = json.loads(i)
            article_tokens = d['article']
            question_tokens = d['question']
            context_tokens = tokenizer.tokenize(article_tokens)  # 文章
            start_ending_tokens = tokenizer.tokenize(question_tokens)  # 问题
            cnt += 1
            choices_list = [d['option_0'], d['option_1'], d['option_2'], d['option_3'], d['option_4']]
            label_list.append(d['label'])
            choices_features = []
            for j in choices_list:
                context_tokens_choice = context_tokens[:]
                question_tokens = re.sub(r'@placeholder', j, question_tokens)
                start_ending_tokens = tokenizer.tokenize(question_tokens)
                ending_token = tokenizer.tokenize(j)  # 选项
                option_len = len(ending_token)
                ques_len = len(start_ending_tokens)
                # ending_tokens = start_ending_tokens + ending_token
                ending_tokens = start_ending_tokens

                _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
                doc_len = len(context_tokens_choice)
                if len(ending_tokens) + len(context_tokens_choice) >= max_seq_length - 3:
                    ques_len = len(ending_tokens) - option_len

                tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
                segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                segment_ids = torch.tensor(segment_ids)
                input_ids = torch.tensor(input_ids)
                input_mask = torch.tensor(input_mask)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                # assert (doc_len + ques_len + option_len) <= max_seq_length
                # if (doc_len + ques_len + option_len) > max_seq_length:
                #     print(doc_len, ques_len, option_len, len(context_tokens_choice), len(ending_tokens))
                #     assert (doc_len + ques_len + option_len) <= max_seq_length

                choices_features.append(
                    {'tokens': tokens,
                     'input_ids': input_ids,
                     'attention_mask': input_mask,
                     'token_type_ids': segment_ids,
                     'position_ids': torch.arange(2, 2 + max_seq_length) if 'Roberta' in str(type(tokenizer)) else torch.arange(max_seq_length),
                     'doc_len': doc_len,
                     'ques_len': ques_len,
                     'option_len': option_len
                     }
                )
            features.append((choices_features, label_list[cnt-1]))

    return features


def load_and_cache_examples_DCMN(args, model_name, tokenizer, set_type='train'):
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
        features = DCMN_get_features_from_ReCAM(data_path,
                                                tokenizer,
                                                args['max_seq_length'], )
        torch.save(features, cached_features_file)
    return features


if __name__ == '__main__':
    import os

    os.chdir('..')

    from transformers import RobertaTokenizer

    model_path = r"D:/transformer_files/roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    train_data = DCMN_get_features_from_ReCAM('datasets/Task2/Task_2_train.jsonl',
                                              tokenizer=tokenizer,
                                              max_seq_length=160, num_choices=5)
