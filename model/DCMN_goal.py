import torch.nn as nn
from torch.autograd import Variable
from transformers.modeling_electra import ElectraConfig, ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST, SequenceSummary
from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel

from utils.functions import *


class FuseNet(nn.Module):
    def __init__(self, config):
        super(FuseNet, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)

    def forward(self, inputs):
        p, q = inputs
        lq = self.linear(q)
        lp = self.linear(p)
        mid = nn.Sigmoid()(lq + lp)
        output = p * mid + q * (1 - mid)
        return output


def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)
    # mask = None
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


class SSingleMatchNet(nn.Module):
    def __init__(self, config):
        super(SSingleMatchNet, self).__init__()
        self.map_linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop_module = nn.Dropout(2 * config.hidden_dropout_prob)
        self.rank_module = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        att_norm = masked_softmax(att_weights, seq_len)

        att_vec = att_norm.bmm(proj_q)
        output = nn.ReLU()(self.trans_linear(att_vec))
        return output


def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i] + 1]
        doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 2:doc_len[i] + ques_len[i] + 2]
        ques_option_seq_output[i, :ques_len[i] + option_len[i]] = sequence_output[i,
                                                                  doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[
                                                                      i] + 1]
        option_seq_output[i, :option_len[i]] = sequence_output[i,
                                               doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[
                                                   i] + 2]
    return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output


class ElectraForMultipleChoiceWithMatch(ElectraPreTrainedModel):
    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sequence_summary = SequenceSummary(config)

        self.classifier2 = nn.Linear(2 * config.hidden_size, 1)
        self.classifier3 = nn.Linear(3 * config.hidden_size, 1)
        self.ssmatch = SSingleMatchNet(config)
        self.fuse = FuseNet(config)

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
            doc_len=None,
            ques_len=None,
            option_len=None,
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        # ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
        # option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()
        outputs = self.electra(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        pooled_output = self.sequence_summary(outputs[0])  # electra输出的[CLS]的向量表示

        doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
            sequence_output, doc_len, ques_len, option_len)

        # pa_output = self.ssmatch([doc_seq_output, option_seq_output, option_len + 1])
        # ap_output = self.ssmatch([option_seq_output, doc_seq_output, doc_len + 1])
        pq_output = self.ssmatch([doc_seq_output, ques_seq_output, ques_len + 1])
        qp_output = self.ssmatch([ques_seq_output, doc_seq_output, doc_len + 1])
        qa_output = self.ssmatch([ques_seq_output, option_seq_output, option_len + 1])
        aq_output = self.ssmatch([option_seq_output, ques_seq_output, ques_len + 1])

        # pa_output_pool, _ = pa_output.max(1)
        # ap_output_pool, _ = ap_output.max(1)
        pq_output_pool, _ = pq_output.max(1)
        qp_output_pool, _ = qp_output.max(1)
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)

        # pa_fuse = self.fuse([pa_output_pool, ap_output_pool])
        pq_fuse = self.fuse([pq_output_pool, qp_output_pool])
        qa_fuse = self.fuse([qa_output_pool, aq_output_pool])

        # cat_pool = torch.cat([pa_fuse, pq_fuse, qa_fuse], 1)
        cat_pool = torch.cat([pq_fuse, qa_fuse], 1)
        output_pool = self.dropout(cat_pool)
        match_logits = self.classifier2(output_pool)
        match_reshaped_logits = match_logits.view(-1, num_choices)

        outputs = (match_reshaped_logits,) + outputs[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(match_reshaped_logits, labels)
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
        doc_len = torch.tensor([j.get('doc_len') for i in x for j in i]).to(device)
        ques_len = torch.tensor([j.get('ques_len') for i in x for j in i]).to(device)
        option_len = torch.tensor([j.get('option_len') for i in x for j in i]).to(device)
        return self.re_forward(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               labels=labels,
                               doc_len=doc_len,
                               ques_len=ques_len,
                               option_len=option_len,
                               head_mask=None, )
