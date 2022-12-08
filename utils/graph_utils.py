import requests
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict
from bidict import bidict
from utils.text_to_uri import standardized_uri
import numpy as np
import re
import string
import urllib.parse
import time
import os


# %%
class GraphUtils:
    def __init__(self):
        self.mp_all = defaultdict(set)  # 记录 id -> id 边的关系
        self.words_to_id = bidict()  # 将一个词映射为 id，仅在 encode_mp 时候构建
        self.words_encode_idx = 0  # 在 encode_mp 时构建
        self.conceptnet_numberbatch_en = dict()
        # 这里之所以不用 GPT2/Roberta tokenizer 是因为空格会被分割为 Ġ
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.args = {
            'n_gram': 3,
            'mp_pickle_path': 'resources/conceptnet5/res_all.pickle',
            'conceptnet_numberbatch_en_path': 'resources/conceptnet5/numberbatch-en.txt',
            'conceptnet_relation_to_id': {
                '/r/AtLocation': 0,
                '/r/CapableOf': 1,
                '/r/Causes': 2,
                '/r/CausesDesire': 3,
                '/r/CreatedBy': 4,
                '/r/DefinedAs': 5,
                '/r/DerivedFrom': 6,
                '/r/Desires': 7,
                '/r/Entails': 8,
                '/r/EtymologicallyDerivedFrom': 9,
                '/r/EtymologicallyRelatedTo': 10,
                '/r/FormOf': 11,
                '/r/HasA': 12,
                '/r/HasContext': 13,
                '/r/HasFirstSubevent': 14,
                '/r/HasLastSubevent': 15,
                '/r/HasPrerequisite': 16,
                '/r/HasProperty': 17,
                '/r/HasSubevent': 18,
                '/r/InstanceOf': 19,
                '/r/IsA': 20,
                '/r/LocatedNear': 21,
                '/r/MadeOf': 22,
                '/r/MannerOf': 23,
                '/r/MotivatedByGoal': 24,
                '/r/NotCapableOf': 25,
                '/r/NotDesires': 26,
                '/r/NotHasProperty': 27,
                '/r/PartOf': 28,
                '/r/ReceivesAction': 29,
                '/r/RelatedTo': 30,
                '/r/SimilarTo': 31,
                '/r/SymbolOf': 32,
                '/r/UsedFor': 33,
            },
            'reduce_noise_args': {
                # 白名单相对优先
                'relation_white_list': [],
                # ['/r/RelatedTo', '/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor', '/r/CapableOf', '/r/AtLocation', '/r/Causes', '/r/HasProperty'],
                'relation_black_list': ['/r/ExternalURL', '/r/Synonym', '/r/Antonym',
                                        '/r/DistinctFrom', '/r/dbpedia/genre', '/r/dbpedia/influencedBy'],
                # 'stop_words': ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an',
                #                'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot',
                #                'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from',
                #                'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however',
                #                'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely',
                #                'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off',
                #                'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says',
                #                'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
                #                'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was', 'we',
                #                'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
                #                'would', 'yet', 'you', 'your'],
                # 'stop_words': ['yours', 'him', 'under', 'from', 'which', 'other', 'been', "that'll", 'here', 'more',
                #                'doesn', 'aren', "couldn't", "aren't", 'did', 'was', "you'll", 'once', 'are', 'in',
                #                'so', 'before', 'not', 'down', 'then', 'ourselves', 'with', 'them', 'how', 'i', 'at',
                #                'up', 'themselves', 'don', "mustn't", 'myself', 'any', 'on', 'it', 'during', 'couldn',
                #                'when', 'do', 'shouldn', 'ain', 'both', 'a', 'there', 'mightn', 'shan', 'whom',
                #                "haven't", "doesn't", 'have', 'had', 'will', 'who', 'about', 'between', 'ours',
                #                'mustn', "you're", 'through', 'he', 'into', 'most', 're', 'by', 'hadn', 'doing',
                #                'they', 'be', 'hers', 'were', 'against', 'nor', "don't", 's', 'yourself', 'each',
                #                "needn't", 'to', 'same', 'these', 'own', "shan't", 'herself', 'after', 't', "mightn't",
                #                'hasn', 'has', 'can', 'only', 'she', 'm', "should've", 'll', 'an', "hasn't", 'the',
                #                'such', 'that', 'and', "hadn't", 'being', 'if', "you've", "didn't", 'now', 'we', 'does',
                #                'won', 'over', 'until', 'yourselves', 'as', 'wouldn', 'or', 'should', 'this', 'is',
                #                'their', 'her', 'me', 'you', 'itself', 'what', 'off', 'where', 'didn', 'my', 'his',
                #                'than', 'needn', 'weren', 'himself', 'very', 'some', 've', 'your', 'those', 'too',
                #                'again', 'of', 'for', 'haven', 'further', "wouldn't", "won't", 'why', 'd', 'am', 'few',
                #                'just', "she's", "shouldn't", 'ma', 'while', 'below', "weren't", 'having', "wasn't",
                #                'because', "you'd", 'theirs', 'y', 'isn', 'all', 'o', "it's", "isn't", 'our', 'wasn',
                #                'its', 'but', 'above', 'out', 'no',
                #                'able', 'across', 'almost', 'also', 'among', 'cannot', 'could', 'dear', 'either', 'else',
                #                'ever', 'every', 'get', 'got', 'however', 'least', 'let', 'like', 'likely', 'may',
                #                'might', 'must', 'neither', 'often', 'rather', 'said', 'say', 'says', 'since', 'tis',
                #                'twas', 'us', 'wants', 'would', 'yet'],
                'stop_words': list(np.load(r'utils/stop_words.npz')['arr_0']),
                'weight_limit': 1.0,
                'edge_count_limit': 100,  # 保留权重最大的 edge_count_limit 条边
            }
        }

    def load_mp_all_by_pickle(self, fpath):
        """
        从 pickle 中加载 conceptnet 图
        :param fpath:
        :return:
        """
        graph_zip = None
        with open(fpath, 'rb') as f:
            graph_zip = pickle.load(f)
        self.mp_all, = graph_zip
        return graph_zip

    def reduce_graph_noise(self, is_overwrite=True):
        """
        基于 relation type 以及 edge weight 降低图的噪声
        :param is_overwrite: 是否写入到 self.mp 中
        :return:
        """
        relation_white_list = self.args['reduce_noise_args']['relation_white_list']
        relation_black_list = self.args['reduce_noise_args']['relation_black_list']
        stop_words = self.args['reduce_noise_args']['stop_words']
        weight_limit = self.args['reduce_noise_args']['weight_limit']
        edge_count_limit = self.args['reduce_noise_args']['edge_count_limit']
        is_black_list = True  # 默认是开启黑名单的

        if len(relation_white_list) != 0:
            # 如果白名单里有则启用白名单
            is_black_list = False

        new_mp = defaultdict(set)  # 记录 id -> id 边的关系
        for key, values in self.mp_all.items():
            st_words = key
            if st_words in stop_words:
                # 停用词跳过
                continue

            # 取 values 按 edge_weight 从大到小排序以后的前 edge_count_limit 个
            to_values = sorted(list(values), key=lambda x: x[2], reverse=True)
            edge_count = 0
            for value in to_values:
                to_words = value[0]
                to_relation = value[1]
                to_weight = value[2]
                if to_words in stop_words:
                    # 停用词跳过
                    continue
                if to_weight < weight_limit:
                    # 边权较低的跳过
                    continue
                if is_black_list:
                    # 如果黑名单开启并且当前 relation 在黑名单里跳过
                    if to_relation in relation_black_list:
                        continue
                else:
                    # 白名单下如果 relation 不在白名单里跳过
                    if to_relation not in relation_white_list:
                        continue
                new_mp[st_words].add((to_words, to_relation, to_weight))
                edge_count += 1
                if edge_count >= edge_count_limit:
                    break

        if is_overwrite:
            self.mp_all = new_mp
        return new_mp

    def merge_graph_by_downgrade(self, is_overwrite=True):
        """
        降级合并 mp 图，将形如 /c/en/apple/n 降级为 /c/en/apple，并省略 /c/en/
        :param is_overwrite: 是否将最终的结果直接写入 self 中的对象
        :return: 降级以后的 mp
        """
        new_mp = defaultdict(set)  # 记录 id -> id 边的关系
        refine_sent = lambda s: re.match('/c/en/([^/]+)', s).group(1)
        for key, values in self.mp_all.items():
            st_words = refine_sent(key)
            for value in values:
                to_words = refine_sent(value[0])
                to_relation = value[1]
                to_weight = value[2]
                new_mp[st_words].add((to_words, to_relation, to_weight))
        if is_overwrite:
            self.mp_all = new_mp
        return new_mp

    def init(self, is_load_necessary_data=True):
        """
        load 部分数据初始化
        :return:
        """
        self.__init__()
        if is_load_necessary_data:
            self.load_mp_all_by_pickle(self.args['mp_pickle_path'])
            # self.load_roberta_numberbatch()  # 加载 roberta 的 embedding
            self.load_conceptnet_numberbatch(self.args['conceptnet_numberbatch_en_path'])

    def get_features_from_words(self, words):
        """
        获取 words 的词向量
        :param words:
        :return:
        """
        words = standardized_uri('en', words).replace('/c/en/', '')
        res = self.conceptnet_numberbatch_en.get(words)
        if res is None:
            """解决 OOV 问题，待定，暂时用 ## 代替"""
            # res = self.conceptnet_numberbatch_en.get('##')
            res = [0.0 for _ in range(len(self.conceptnet_numberbatch_en.get('0')))]
            # print('OOV fail!!! ', words)
        # else:
        # print('OOV ssc ', words)
        return res

    def load_roberta_numberbatch(self, fpath='conceptnet5/numberbatch_roberta_fast.pickle'):
        # 加载 Roberta 编码的词向量
        if len(self.conceptnet_numberbatch_en) != 0:
            # 如果已经加载则不用管了
            return
        self.conceptnet_numberbatch_en.clear()
        res = []
        with open(fpath, 'rb') as f:
            res = pickle.load(f)
        print('load roberta numberbatch: ', len(res))
        for word, embedding in res:
            self.conceptnet_numberbatch_en[word] = embedding
        print('load roberta numberbatch done!')

    def get_default_oov_feature(self):
        # 默认的 oov feature
        return [0.0 for _ in range(300)]

    def load_conceptnet_numberbatch(self, fpath):
        """
        从 numberbatch 中加载每个词的词向量
        :param fpath:
        :return:
        """
        if len(self.conceptnet_numberbatch_en) != 0:
            # 如果已经加载则不用管了
            return
        self.conceptnet_numberbatch_en.clear()
        with open(fpath, 'r', encoding='UTF-8') as f:
            n, vec_size = list(map(int, f.readline().split(' ')))
            print('load conceptnet numberbatch: ', n, vec_size)
            for i in range(n):
                tmp_data = f.readline().split(' ')
                words = str(tmp_data[0])
                vector = list(map(float, tmp_data[1:]))
                self.conceptnet_numberbatch_en[words] = vector
            print('load conceptnet numberbatch done!')

    def get_words_from_id(self, id):
        """
        从 id 获取对应的 words
        :param id:
        :return:
        """
        return self.words_to_id.inverse.get(id)

    def get_id_from_words(self, words, is_add=False):
        """
        从 words 获取其映射的 id
        :param words:
        :param is_add: 如果不存在是否加入进去
        :return:
        """
        if self.words_to_id.get(words) is None and is_add:
            self.words_to_id[words] = self.words_encode_idx
            self.words_encode_idx += 1
        return self.words_to_id.get(words)

    def encode_index(self, mp):
        """
        建立一个 words to id 的映射表，使用 bidict 可以双向解析
        """
        x_index_id = []
        edge_index = []
        edge_weight = []
        edge_type = []
        self.words_encode_idx = 0
        self.words_to_id.clear()
        # 建边
        for key, values in mp.items():
            st_id = self.get_id_from_words(key, is_add=True)
            x_index_id.append(st_id)  # 代表所有出现在 sentence 中的 word id
            for value in values:
                to_words = value[0]
                to_relation = value[1]
                to_weight = value[2]

                ed_id = self.get_id_from_words(to_words, is_add=True)

                # 暂定建立双向边
                edge_index.append([st_id, ed_id])
                edge_weight.append(to_weight)
                edge_type.append(self.args['conceptnet_relation_to_id'].get(to_relation))
                edge_index.append([ed_id, st_id])
                edge_weight.append(to_weight)
                edge_type.append(self.args['conceptnet_relation_to_id'].get(to_relation))
        # 建立每个 id 对应词与词向量映射关系
        x = [self.get_features_from_words(self.get_words_from_id(i)) for i in range(self.words_encode_idx)]
        x_index = torch.zeros(len(x), dtype=torch.bool)
        x_index[x_index_id] = 1
        return torch.tensor(x), x_index, torch.tensor(edge_index).t(), torch.tensor(edge_weight)

    def get_submp_by_sentences(self, sentences: list, is_merge=False):
        """
        获取 conceptnet 中的一个子图
        :param sentences: 一个列表，如 ["I am a student.", "Hi!"]
        :param is_merge: 是否合并 sentences 中每个元素的结果
        :return: 子图 mp
        """

        def get_submp_by_one(mp_all, sent, n_gram=1, stop_words=[]):
            mp_sub = defaultdict(set)
            sent = sent.strip(',|.|?|;|:|!').lower()
            tokens = self.tokenizer.tokenize(sent)
            # 这里加 '#' 是为了防止 tokenizer 与传统根据分割方法 n_gram 在一起
            tokens += ['#'] + re.sub('[^a-zA-Z0-9,]', ' ', sent).split(' ')
            for gram in range(1, n_gram + 1):
                start, end = 0, gram
                while end <= len(tokens):
                    # n_gram 将要连接的单词，中间需要用 _ 分割
                    q_words = '_'.join(tokens[start:end])
                    start, end = start + 1, end + 1

                    if gram == 1 and q_words in stop_words:
                        # 如果 q_words 是停用词
                        continue
                    if q_words.find('#') != -1:
                        # 如果其中有 # 号
                        continue

                    if mp_all.get(q_words) is not None and mp_sub.get(q_words) is None:
                        # 如果 mp_all 中存在 q_words 并且 mp_sub 中不存在 q_words
                        mp_sub[q_words] |= mp_all[q_words]
            return mp_sub

        if is_merge:
            sent = ' '.join(sentences)
            sentences = [sent]

        res = []
        for i in sentences:
            res.append(get_submp_by_one(self.mp_all, i, self.args['n_gram'],
                                        stop_words=self.args['reduce_noise_args']['stop_words']))
        return res


# %%
if __name__ == '__main__':
    from utils.getGraphUtils import get_datas, get_data_from_task_2
    import os

    os.chdir('..')
    graph = GraphUtils()
    graph.init(is_load_necessary_data=True)
    graph.merge_graph_by_downgrade()
    mp = graph.reduce_graph_noise(is_overwrite=False)
    # data = np.array(get_datas(
    #     get_data_from_task_2(
    #         '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training  Data/subtaskB_data_all.csv',
    #         '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training  Data/subtaskB_answers_all.csv'),
    #     get_data_from_task_2(
    #         '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv',
    #         '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv')
    # ))
    # mp = graph.get_submp_by_sentences(['I am a students.', 'You have a apples and bananas'], is_merge=True)[0]
    # x, edge_index, edge_weight = graph.encode_index(mp)
