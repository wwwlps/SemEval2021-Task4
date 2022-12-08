import torch
import torch.nn as nn
from transformers.modeling_roberta import RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, RobertaModel
from transformers.modeling_electra import ElectraConfig, ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST, ElectraModel, \
    SequenceSummary, ElectraPreTrainedModel
from transformers.modeling_bert import BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_LIST, BertModel, \
    BertPreTrainedModel
from transformers.modeling_albert import AlbertConfig, ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST, AlbertModel, \
    AlbertPreTrainedModel
from utils.functions import *

# def GassianKenerl(X, gamma):


class SVM(nn.Module):
    def __init__(self, in_features_size):
        super(SVM, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features_size))
        self.bias = nn.Parameter(torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.5, 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, X, gamma):
        value = torch.exp(torch.sum(torch.mul(X - self.weight, X - self.weight), dim=1) / (-2 * gamma * gamma)) - self.bias
        print(self.weight)
        print(torch.mul(X - self.weight, X - self.weight))
        print(torch.sum(torch.mul(X - self.weight, X - self.weight), dim=1) / (-2 * gamma * gamma))
        return value


class ElectraForMultipleChoice(ElectraPreTrainedModel):
    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, is_add_projection=False):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sequence_summary = SequenceSummary(config)
        self.is_projection = False
        if is_add_projection:
            self.is_projection = True
            self.project1 = nn.Linear(config.hidden_size, config.hidden_size)  # 添加的变换层
            self.project2 = nn.Linear(config.hidden_size, config.hidden_size)  # 添加的变换层
        # self.DNN = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        # )

        # self.SVM = SVM(config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def re_forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # print(attention_mask.shape)
        if attention_mask is not None and attention_mask.dim() == 3:
            flat_attention_mask = attention_mask.view(-1,
                                                      attention_mask.size(-1)) if attention_mask is not None else None
        else:
            flat_attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None
        outputs = self.electra(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        # pooled_output = outputs[0][:, 0]
        pooled_output = self.sequence_summary(outputs[0])   # electra输出的[CLS]的向量表示, [batch_size*5, hidden_size]
        # pooled_output = self.dropout(pooled_output)
        '''
        变换模块，用来把bert输出的向量投射到另外一个空间中去，再进入到后面的classifier
        '''
        if self.is_projection:
            pooled_output = self.project1(pooled_output)
            pooled_output = torch.relu(pooled_output)  # 可以用不同的非线性激活函数
            pooled_output = self.dropout(pooled_output)

            # pooled_output = self.project2(pooled_output)
            # pooled_output = gelu(pooled_output)  # 可以用不同的非线性激活函数

        # pooled_output = self.DNN(pooled_output)
        # print(pooled_output)
        # logits = self.SVM(pooled_output, 1000)
        # print(logits)
        # assert 1==2
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def forward(self, x, labels=None):
        num_choices = len(x[0])
        device = labels.device
        # input_ids = torch.stack([j.get('input_ids') for i in x for j in i], dim=0).reshape(
        #     -1, num_choices, x[0][0].get('input_ids').size(0)).to(device)
        # token_type_ids = torch.stack([j.get('token_type_ids') for i in x for j in i], dim=0).reshape(
        #     -1, num_choices, x[0][0].get('token_type_ids').size(0)).to(device)
        # attention_mask = torch.stack([j.get('attention_mask') for i in x for j in i], dim=0).reshape(
        #     -1, num_choices, x[0][0].get('attention_mask').size(0)).to(device)
        # position_ids = torch.stack([j.get('position_ids') for i in x for j in i], dim=0).reshape(
        #     -1, num_choices, x[0][0].get('position_ids').size(0)).to(device)
        # print(x[0][0].get('attention_mask').shape)
        # print(x[0][0].get('attention_mask').size(0))
        input_ids = torch.stack([j.get('input_ids') for i in x for j in i], dim=0).reshape(
            (-1, num_choices) + x[0][0].get('input_ids').shape).to(device)
        token_type_ids = torch.stack([j.get('token_type_ids') for i in x for j in i], dim=0).reshape(
            (-1, num_choices) + x[0][0].get('token_type_ids').shape).to(device)
        attention_mask = torch.stack([j.get('attention_mask') for i in x for j in i], dim=0).reshape(
            (-1, num_choices) + x[0][0].get('attention_mask').shape).to(device)
        position_ids = torch.stack([j.get('position_ids') for i in x for j in i], dim=0).reshape(
            (-1, num_choices) + x[0][0].get('position_ids').shape).to(device)
        return self.re_forward(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               labels=labels,
                               head_mask=None)


class ElectraForMultipleChoicePro(ElectraPreTrainedModel):
    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, is_add_projection=False):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sequence_summary = SequenceSummary(config)
        self.is_projection = False
        if is_add_projection:
            self.is_projection = True
            self.project1 = nn.Linear(config.hidden_size, config.hidden_size+500)  # 添加的变换层
            # self.project2 = nn.Linear(config.hidden_size, config.hidden_size)  # 添加的变换层
        # self.DNN = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        # )

        # self.SVM = SVM(config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size+500, 1)

        self.init_weights()

    def re_forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.electra(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        # pooled_output = outputs[0][:, 0]
        pooled_output = self.sequence_summary(outputs[0])   # electra输出的[CLS]的向量表示, [batch_size*5, hidden_size]
        # pooled_output = self.dropout(pooled_output)
        '''
        变换模块，用来把bert输出的向量投射到另外一个空间中去，再进入到后面的classifier
        '''
        if self.is_projection:
            pooled_output = self.project1(pooled_output)
            pooled_output = torch.relu(pooled_output)  # 可以用不同的非线性激活函数
            pooled_output = self.dropout(pooled_output)

            # pooled_output = self.project2(pooled_output)
            # pooled_output = gelu(pooled_output)  # 可以用不同的非线性激活函数

        # pooled_output = self.DNN(pooled_output)
        # print(pooled_output)
        # logits = self.SVM(pooled_output, 1000)
        # print(logits)
        # assert 1==2
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def forward(self, x, labels=None):
        num_choices = len(x[0])
        device = labels.device
        input_ids = torch.stack([j.get('input_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('input_ids').size(0)).to(device)
        token_type_ids = torch.stack([j.get('token_type_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('token_type_ids').size(0)).to(device)
        attention_mask = torch.stack([j.get('attention_mask') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('attention_mask').size(0)).to(device)
        position_ids = torch.stack([j.get('position_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('position_ids').size(0)).to(device)
        return self.re_forward(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               labels=labels,
                               head_mask=None)


class ElectraForMultipleChoiceHard(ElectraPreTrainedModel):
    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, is_add_projection=False):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def re_forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.electra(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = self.sequence_summary(outputs[0])   # electra输出的[CLS]的向量表示
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def forward(self, x, labels=None):
        num_choices = len(x[0])
        device = labels.device
        input_ids = torch.stack([j.get('input_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('input_ids').size(0)).to(device)
        token_type_ids = torch.stack([j.get('token_type_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('token_type_ids').size(0)).to(device)
        attention_mask = torch.stack([j.get('attention_mask') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('attention_mask').size(0)).to(device)
        position_ids = torch.stack([j.get('position_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('position_ids').size(0)).to(device)
        return self.re_forward(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               labels=labels,
                               head_mask=None)


class ElectraForMultipleChoiceBinary(ElectraPreTrainedModel):
    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, is_add_projection=False):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def re_forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.electra(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = self.sequence_summary(outputs[0])   # electra输出的[CLS]的向量表示
        # pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, 2)

        outputs = (reshaped_logits,) + outputs[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def forward(self, x, labels=None):
        device = labels.device
        input_ids = torch.stack([j.get('input_ids') for i in x for j in i], dim=0).reshape(
            -1, x[0][0].get('input_ids').size(0)).to(device)
        token_type_ids = torch.stack([j.get('token_type_ids') for i in x for j in i], dim=0).reshape(
            -1, x[0][0].get('token_type_ids').size(0)).to(device)
        attention_mask = torch.stack([j.get('attention_mask') for i in x for j in i], dim=0).reshape(
            -1, x[0][0].get('attention_mask').size(0)).to(device)
        position_ids = torch.stack([j.get('position_ids') for i in x for j in i], dim=0).reshape(
            -1, x[0][0].get('position_ids').size(0)).to(device)
        return self.re_forward(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               labels=labels,
                               head_mask=None)


class AlbertForMultipleChoice(AlbertPreTrainedModel):
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def re_forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.albert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def forward(self, x, labels=None):
        num_choices = len(x[0])
        device = labels.device
        input_ids = torch.stack([j.get('input_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('input_ids').size(0)).to(device)
        token_type_ids = torch.stack([j.get('token_type_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('token_type_ids').size(0)).to(device)
        attention_mask = torch.stack([j.get('attention_mask') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('attention_mask').size(0)).to(device)
        position_ids = torch.stack([j.get('position_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('position_ids').size(0)).to(device)
        return self.re_forward(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               labels=labels,
                               head_mask=None)


class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config, is_add_projection=False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.is_projection = False
        if is_add_projection:
            self.is_projection = True
            self.project1 = nn.Linear(config.hidden_size, config.hidden_size)  # 添加的变换层
            self.project2 = nn.Linear(config.hidden_size, config.hidden_size)  # 添加的变换层

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def re_forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        '''
        变换模块，用来把bert输出的向量投射到另外一个空间中去，再进入到后面的classifier
        '''
        if self.is_projection:
            pooled_output = self.project1(pooled_output)
            pooled_output = torch.relu(pooled_output)  # 可以用不同的非线性激活函数
            pooled_output = self.dropout(pooled_output)

            # pooled_output = self.project2(pooled_output)
            # pooled_output = gelu(pooled_output)  # 可以用不同的非线性激活函数

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        # print(reshaped_logits.shape[0])
        # print(len(labels))

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def forward(self, x, labels=None):
        num_choices = len(x[0])
        device = labels.device
        input_ids = torch.stack([j.get('input_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('input_ids').size(0)).to(device)
        token_type_ids = torch.stack([j.get('token_type_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('token_type_ids').size(0)).to(device)
        attention_mask = torch.stack([j.get('attention_mask') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('attention_mask').size(0)).to(device)
        position_ids = torch.stack([j.get('position_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('position_ids').size(0)).to(device)

        return self.re_forward(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               labels=labels,
                               head_mask=None)


class BertForMultipleChoice(BertPreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def re_forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.bert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def forward(self, x, labels=None):
        num_choices = len(x[0])
        device = labels.device
        input_ids = torch.stack([j.get('input_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('input_ids').size(0)).to(device)
        token_type_ids = torch.stack([j.get('token_type_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('token_type_ids').size(0)).to(device)
        attention_mask = torch.stack([j.get('attention_mask') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('attention_mask').size(0)).to(device)
        position_ids = torch.stack([j.get('position_ids') for i in x for j in i], dim=0).reshape(
            -1, num_choices, x[0][0].get('position_ids').size(0)).to(device)
        return self.re_forward(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               labels=labels,
                               head_mask=None)
