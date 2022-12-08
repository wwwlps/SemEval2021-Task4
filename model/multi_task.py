import torch
import torch.nn as nn
from transformers.modeling_electra import ElectraConfig, ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST, ElectraModel, \
    SequenceSummary, ElectraPreTrainedModel


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels, do_not_summary):
        super(ElectraClassificationHead, self).__init__()
        self.do_not_summary = do_not_summary
        if not do_not_summary:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        if not self.do_not_summary:
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraForMultipleChoice_Mul(ElectraPreTrainedModel):
    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, task_output_config, do_not_summary, same_linear_layer):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.project = nn.Linear(config.hidden_size, config.hidden_size)
        # self.classifier = [nn.Linear(config.hidden_size, 1)]
        self.classifiers = []
        self.num_labels = task_output_config
        self.task_id_map = {}

        if same_linear_layer:
            multi_choice_index = None
            for idx, num_label in enumerate(task_output_config):
                if multi_choice_index is not None:
                    self.task_id_map[idx] = multi_choice_index
                else:
                    self.classifiers.append(ElectraClassificationHead(config, 1, do_not_summary))
                    self.task_id_map[idx] = idx
                    multi_choice_index = idx
        else:
            for idx, num_label in enumerate(task_output_config):
                self.classifiers.append(ElectraClassificationHead(config, 1, do_not_summary))
                self.task_id_map[idx] = idx
        self.classifiers = nn.ModuleList(self.classifiers)
        self.init_weights()

    def re_forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            position_ids=None,
            head_mask=None,
            task_id=None,  # task编号
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

        logits = self.classifiers[self.task_id_map[task_id]](outputs[0])
        reshaped_logits = logits.view(-1, self.num_labels[task_id])

        outputs = (reshaped_logits,) + outputs[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

    def forward(self, x, labels=None, task_id=0):
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
                               head_mask=None,
                               task_id=task_id,)
