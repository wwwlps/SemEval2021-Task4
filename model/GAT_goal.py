from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel, AlbertConfig
from transformers import RobertaConfig, RobertaModel
from transformers.modeling_albert import AlbertPreTrainedModel, ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST, \
    AlbertEmbeddings, AlbertTransformer
from transformers.modeling_roberta import RobertaLMHead, ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers import GPT2DoubleHeadsModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers.modeling_electra import ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST, \
    ElectraConfig, ElectraModel, ElectraPreTrainedModel, SequenceSummary
from utils.loss import FocalLoss
from utils.functions import gelu


class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        from torch_geometric.nn import GCNConv, GATConv, GINConv
        # nn1 = nn.Sequential(
        #     nn.Linear(300, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # nn2 = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # self.conv1 = GINConv(nn1)
        # self.conv2 = GINConv(nn2)
        self.conv1 = GATConv(300, 128)
        self.conv2 = GATConv(128, 128)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, data):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x, edge_index, edge_weight = data.x, data.edge_index, None  # for GAT
        x = self.conv1(x, edge_index, edge_weight)
        x = gelu(x)
        x = F.dropout(x, training=self.training)
        logits = self.conv2(x, edge_index, edge_weight)
        logits = torch.stack(
            [logits[data.batch == i][data.pos[data.batch == i]].mean(dim=0) for i in range(data.num_graphs)], dim=0)

        x = gelu(logits)
        x = self.fc1(x)

        outputs = (x.reshape(-1, data.num_graphs), logits,)

        if data.y is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs[0], data.y.reshape(-1, data.num_graphs).argmax(dim=1))
            outputs = (loss,) + outputs

        return outputs


class ElectraForMultipleChoice_GAT(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        pre_input_ids = input_ids
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # if attention_mask.dim() == 3:
        #     attention_mask = attention_mask.view(-1,
        #                                               attention_mask.size(-1)) if attention_mask is not None else None
        # else:
        #     attention_mask = attention_mask.view(
        #         (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        sequence_output = discriminator_hidden_states[0]

        pooled_output = self.summary(sequence_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits, pooled_output.view(pre_input_ids.shape[0], num_choices, -1),) + discriminator_hidden_states[
            1:
        ]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class GAT_goal_model(nn.Module):
    def __init__(self, args, is_add_projection=False):
        super(GAT_goal_model, self).__init__()
        self.args = args
        # roberta_config = AlbertConfig.from_pretrained('albert-base-v2')
        # self.roberta = AlbertForMultipleChoice.from_pretrained(
        #     'pre_weights/albert-base-v2-pytorch_model.bin', config=roberta_config)
        # roberta_config = RobertaConfig.from_pretrained('roberta-large')
        # roberta_config.attention_probs_dropout_prob = 0.2
        # roberta_config.hidden_dropout_prob = 0.2

        # if args.get('with_lm'):
        #     self.roberta = RobertaForMultipleChoiceWithLM.from_pretrained(
        #         'pre_weights/roberta-large_model.bin', config=roberta_config)
        # else:
        #     self.roberta = RobertaForMultipleChoice.from_pretrained(
        #         'pre_weights/roberta-large_model.bin', config=roberta_config)
        electra_large_model_path = args['pretrained_model_path'] + 'Electra-large'
        electra_config = ElectraConfig.from_pretrained(electra_large_model_path)
        electra_config.attention_probs_dropout_prob = 0.2
        electra_config.hidden_dropout_prob = 0.2
        self.electra = ElectraForMultipleChoice_GAT.from_pretrained(electra_large_model_path)

        from utils.attentionUtils import SelfAttention
        self.gcn = GCNNet()
        self.merge_fc1 = nn.Linear(electra_config.hidden_size + 128, 512)
        self.attn = SelfAttention(512, 8)
        # self.roberta_fc1 = nn.Linear(roberta_config.hidden_size, 128)  # 将 roberta vector 降维到与 gcn 相同
        # self.gcn_fc1 = nn.Linear(128, 128)  # 同上
        self.fc3 = nn.Linear(512 + electra_config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

        self.is_projection = False
        if is_add_projection:
            self.is_projection = True
            self.project = nn.Linear(512 + electra_config.hidden_size, 512 + electra_config.hidden_size)  # 增加的MLP变换层

    def forward(self, x, labels=None):
        semantic_features = [i[0] for i in x]
        num_choices = len(semantic_features[0])
        input_ids = torch.stack([j.get('input_ids') for i in semantic_features for j in i], dim=0).reshape(
            -1, num_choices, semantic_features[0][0].get('input_ids').size(0)).to(
            self.args['device'])
        attention_mask = torch.stack([j.get('attention_mask') for i in semantic_features for j in i], dim=0).reshape(
            -1, num_choices, semantic_features[0][0].get('attention_mask').size(0)).to(
            self.args['device'])
        token_type_ids = torch.stack([j.get('token_type_ids') for i in semantic_features for j in i], dim=0).reshape(
            -1, num_choices, semantic_features[0][0].get('token_type_ids').size(0)).to(self.args['device'])
        position_ids = torch.stack([j.get('position_ids') for i in semantic_features for j in i], dim=0).reshape(
            -1, num_choices, semantic_features[0][0].get('position_ids').size(0)).to(
            self.args['device'])

        graph_features = [i[1].to(self.args['device']) for i in x]
        labels = labels.to(self.args['device'])

        gcn_tmp_features = [self.gcn(i) for i in graph_features]

        electra_outputs = self.electra(input_ids,
                                       attention_mask=attention_mask,
                                       # token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)

        graph_features = [i[1].to('cpu') for i in x]

        loss = electra_outputs[0]  # roberta loss
        # electra reshaped_logits
        # print(len(electra_outputs))
        electra_logits = electra_outputs[2]

        loss = loss + torch.stack([i[0] for i in gcn_tmp_features]).mean()  # + gcn loss
        gcn_features = torch.stack([i[2] for i in gcn_tmp_features])  # [4, 3, 64]
        del gcn_tmp_features, electra_outputs  # 清理显存

        merge_features = self.merge_fc1(
            torch.cat((electra_logits, gcn_features), dim=2))
        merge_features = self.attn(merge_features)[0]

        # roberta_logits = self.roberta_fc1(roberta_logits)
        # gcn_features = self.gcn_fc1(gcn_features)
        # merge_features = roberta_logits + gcn_features

        # roberta_logits 最后是 tanH 算出来的，这里用 gelu 好不好
        # merge_features = nn.Tanh()(merge_features)
        merge_features = gelu(merge_features)
        merge_features = self.dropout(merge_features)
        """
        加个变换模块, 非线性激活
        """
        if self.is_projection:
            merge_features = self.project(torch.cat((electra_logits, merge_features), dim=2))
            merge_features = torch.relu(merge_features)  # 这里的激活函数可以变换
            merge_features = self.fc3(merge_features).view(-1, num_choices)
        else:
            merge_features = self.fc3(torch.cat((electra_logits, merge_features), dim=2)).view(-1, num_choices)
            # merge_features = (self.fc3(merge_features) + self.fc3(roberta_logits)).view(-1, num_choices)

        outputs = merge_features,

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss + loss_fct(outputs[0], labels)  # merge loss

            outputs = (loss,) + outputs
        return outputs

