import numpy as np
# import dgl


def add_knowledge_with_vm(mp_all,
                          sent_batch,
                          tokenizer,
                          article,
                          question,
                          option,
                          max_entities=2,
                          max_length=128):
    """
    input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
    return: know_sent_batch - list of sentences with entites embedding
            position_batch - list of position index of each character.
            visible_matrix_batch - list of visible matrixs
            seg_batch - list of segment tags
    """

    def conceptnet_relation_to_nl(ent):
        """
        :param ent: ('university', '/r/AtLocation', 6.325)
        :return: 返回 ent 翻译成自然语言并分词后的结果
        """
        relation_to_language = {'/r/AtLocation': 'is at the location of the',
                                '/r/CapableOf': 'is capable of',
                                '/r/Causes': 'causes',
                                '/r/CausesDesire': 'causes the desire of',
                                '/r/CreatedBy': 'is created by',
                                '/r/DefinedAs': 'is defined as',
                                '/r/DerivedFrom': 'is derived from',
                                '/r/Desires': 'desires',
                                '/r/Entails': 'entails',
                                '/r/EtymologicallyDerivedFrom': 'is etymologically derived from',
                                '/r/EtymologicallyRelatedTo': 'is etymologically related to',
                                '/r/FormOf': 'is an inflected form of',
                                '/r/HasA': 'has a',
                                '/r/HasContext': 'appears in the context of',
                                '/r/HasFirstSubevent': 'is an event that begins with subevent',
                                '/r/HasLastSubevent': 'is an event that concludes with subevent',
                                '/r/HasPrerequisite': 'has prerequisite is',
                                '/r/HasProperty': 'has an attribute is',
                                '/r/HasSubevent': 'has a subevent is',
                                '/r/InstanceOf': 'runs an instance of',
                                '/r/IsA': 'is a',
                                '/r/LocatedNear': 'is located near',
                                '/r/MadeOf': 'is made of',
                                '/r/MannerOf': 'is the manner of',
                                '/r/MotivatedByGoal': 'is a step toward accomplishing the goal',
                                '/r/NotCapableOf': 'is not capable of',
                                '/r/NotDesires': 'does not desire',
                                '/r/NotHasProperty': 'has no attribute',
                                '/r/PartOf': 'is a part of',
                                '/r/ReceivesAction': 'receives action for',
                                '/r/RelatedTo': 'is related to',
                                '/r/SimilarTo': 'is similar to',
                                '/r/SymbolOf': 'is the symbol of',
                                '/r/UsedFor': 'is used for',
                                }
        # 这里加入一个 i，主要是为了让后面的作为非开头出现
        ent_values = 'i {}'.format(ent[0].replace('_', ' '))
        # ent_values = 'i {} {}'.format(relation_to_language.get(ent[1], ''),
        #                               ent[0].replace('_', ' '))
        ent_values = tokenizer.tokenize(ent_values)[1:]

        # is_bpe_tokenizer = tokenizer.cls_token == '<s>'  # 适用于 Roberta/GPT
        # if is_bpe_tokenizer:
        #     # 因为这里是分支节点，因此是有空格分割的，针对 BPE 算法的分词器应该添加 Ġ
        #     ent_values[0] = 'Ġ' + ent_values[0]
        return ent_values

    split_sent_batch = [tokenizer.tokenize(sent) for sent in sent_batch]
    know_sent_batch = []
    position_batch = []
    visible_matrix_batch = []
    seg_batch = []
    passage_len = len(tokenizer.tokenize(article)) + 2   # CLS + article + SEP
    for split_sent in split_sent_batch:

        # create tree
        sent_tree = []
        pos_idx_tree = []  # soft (主干结点,分支结点)的列表
        abs_idx_tree = []  # hard (主干结点,分支结点)的列表
        pos_idx = -1  # soft position idx，深度相同的节点 idx 相等
        abs_idx = -1  # hard position idx，不重复
        abs_idx_src = []  # 主干上的hard position列表
        cnt = 0
        for token in split_sent:
            """
            k-bert 这里只挑了前 max_entities 个 kg 里邻接的实体，如果采样得出或根据其他方法会不会更好
            """
            # entities = list(mp_all.get(token,
            #                            []))[:max_entities]
            # Ġ 是 GPT-2/Roberta Tokenizer，▁ 是 Albert 中的
            # if cnt >= passage_len and token.strip(',|.|?|;|:|!|Ġ|_|▁') == option:
            if cnt >= passage_len and token.strip(',|.|?|;|:|!|Ġ|_|▁') != option:
            # if cnt < passage_len:
                entities = sorted(list(mp_all.get(token.strip(',|.|?|;|:|!|Ġ|_|▁'), [])), key=lambda x: x[2],
                                  reverse=True)[
                           :max_entities]
            else:
                entities = []
            cnt += 1
            sent_tree.append((token, entities))

            if token in tokenizer.all_special_tokens:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
            else:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
                # token_pos_idx = [
                #     pos_idx + i for i in range(1,
                #                                len(token) + 1)
                # ]
                # token_abs_idx = [
                #     abs_idx + i for i in range(1,
                #                                len(token) + 1)
                # ]
            abs_idx = token_abs_idx[-1]  # 取当前最后一个hard的下标

            entities_pos_idx = []
            entities_abs_idx = []
            for ent in entities:
                ent_values = conceptnet_relation_to_nl(ent)

                ent_pos_idx = [
                    token_pos_idx[-1] + i for i in range(1,
                                                         len(ent_values) + 1)
                ]
                entities_pos_idx.append(ent_pos_idx)
                ent_abs_idx = [abs_idx + i for i in range(1, len(ent_values) + 1)]
                abs_idx = ent_abs_idx[-1]  # 更新hard下标
                entities_abs_idx.append(ent_abs_idx)

            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx

        # Get know_sent and pos
        know_sent = []  # 每个 token 占一个
        pos = []  # 每个 token 的 soft position idx
        seg = []  # token 是属于主干还是分支，主干为 0，分支为 1
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            if word in tokenizer.all_special_tokens:
                know_sent += [word]
                seg += [0]
            else:
                know_sent += [word]
                seg += [0]
            pos += pos_idx_tree[i][0]
            for j in range(len(sent_tree[i][1])):
                ent = sent_tree[i][1][j]  # ('university', '/r/AtLocation', 6.325)
                ent_values = conceptnet_relation_to_nl(ent)

                add_word = ent_values
                know_sent += add_word
                seg += [1] * len(add_word)
                pos += list(pos_idx_tree[i][1][j])

        token_num = len(know_sent)

        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                # abs_idx_src 代表所有主干上的节点 id，src_ids 为当前遍历主干 token 的 id
                # 这里 visible_abs_idx 代表主干上的节点可以看到主干其他节点，并且也可以看到其下面分支的节点
                visible_abs_idx = abs_idx_src + [
                    idx for ent in item[1] for idx in ent
                ]
                visible_matrix[id, visible_abs_idx] = 1
            for ent in item[1]:
                for id in ent:
                    # 这里遍历分支节点，它可以看到该分支上所有节点以及其依赖的那些主干节点
                    # 依赖的主干节点可能有多个，因为一个词比如 “我的世界” 它分字后有四个节点
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [tokenizer.pad_token] * pad_num
            seg += [0] * pad_num
            pos += [max_length - 1] * pad_num
            visible_matrix = np.pad(visible_matrix,
                                    ((0, pad_num), (0, pad_num)),
                                    'constant')  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]

        know_sent_batch.append(know_sent)
        position_batch.append(pos)
        visible_matrix_batch.append(visible_matrix)
        seg_batch.append(seg)

    return know_sent_batch, position_batch, visible_matrix_batch, seg_batch
