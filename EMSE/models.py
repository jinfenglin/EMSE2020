import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, AutoModelForSequenceClassification, BertModel
import torch.nn.functional as F


class TBERT(PreTrainedModel):
    def get_nl_tokenizer(self):
        raise NotImplementedError

    def get_pl_tokenizer(self):
        raise NotImplementedError

    def create_nl_embd(self, input_ids, attention_mask):
        raise NotImplementedError

    def create_pl_embd(self, input_ids, attention_mask):
        raise NotImplementedError

    def get_nl_sub_model(self):
        raise NotImplementedError

    def get_pl_sub_model(self):
        raise NotImplementedError


class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class RelationClassifyHeader(nn.Module):
    """
    H2:
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.code_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)

        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class TBertTwin(TBERT):
    def __init__(self, config, code_bert):
        super().__init__(config)
        # nbert_model = "huggingface/CodeBERTa-small-v1"
        cbert_model = code_bert
        nbert_model = code_bert
        if code_bert == 'microsoft/Multilingual-MiniLM-L12-H384':
            self.ctokneizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.cbert = AutoModel.from_pretrained(cbert_model)

            self.ntokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.nbert = AutoModel.from_pretrained(nbert_model)
        else:
            self.ctokneizer = AutoTokenizer.from_pretrained(cbert_model)
            self.cbert = AutoModel.from_pretrained(cbert_model)

            self.ntokenizer = AutoTokenizer.from_pretrained(nbert_model)
            self.nbert = AutoModel.from_pretrained(nbert_model)

        self.cls = RelationClassifyHeader(config)

    def forward(
            self,
            code_ids=None,
            code_attention_mask=None,
            text_ids=None,
            text_attention_mask=None,
            relation_label=None):
        c_hidden = self.cbert(code_ids, attention_mask=code_attention_mask)[0]
        n_hidden = self.nbert(text_ids, attention_mask=text_attention_mask)[0]

        logits = self.cls(code_hidden=c_hidden, text_hidden=n_hidden)
        output_dict = {"logits": logits}
        if relation_label is not None:
            loss_fct = CrossEntropyLoss()
            rel_loss = loss_fct(logits.view(-1, 2), relation_label.view(-1))
            output_dict['loss'] = rel_loss
        return output_dict  # (rel_loss), rel_score

    def get_sim_score(self, text_hidden, code_hidden):
        logits = self.cls(text_hidden=text_hidden, code_hidden=code_hidden)
        sim_scores = torch.softmax(logits, 1).data.tolist()
        return [x[1] for x in sim_scores]

    def get_nl_tokenizer(self):
        return self.ntokenizer

    def get_pl_tokenizer(self):
        return self.ctokneizer

    def create_nl_embd(self, input_ids, attention_mask):
        return self.nbert(input_ids, attention_mask)

    def create_pl_embd(self, input_ids, attention_mask):
        return self.cbert(input_ids, attention_mask)

    def get_nl_sub_model(self):
        return self.nbert

    def get_pl_sub_model(self):
        return self.cbert


class CosineTrainHeader(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.code_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)
        self.margin = 0.5

    def forward(self, text_hidden, pos_code_hidden, neg_code_hidden):
        pool_pos_code_hidden = self.code_pooler(pos_code_hidden)
        pool_neg_code_hidden = self.code_pooler(neg_code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)

        anchor_sim = F.cosine_similarity(pool_text_hidden, pool_pos_code_hidden)
        neg_sim = F.cosine_similarity(pool_text_hidden, pool_neg_code_hidden)
        loss = (self.margin - anchor_sim + neg_sim).clamp(min=1e-6).mean()
        return loss, anchor_sim, neg_sim


class TBertSiamese(TBertTwin):
    def __init__(self, config, code_bert):
        super().__init__(config, code_bert)
        self.ntokenizer = self.ctokneizer
        self.nbert = self.cbert


class TBertSingle(PreTrainedModel):
    def __init__(self, config, code_bert):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(code_bert)
        self.bert = AutoModelForSequenceClassification.from_pretrained(code_bert)

    def forward(self, input_ids, attention_mask, token_type_ids, relation_label=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=relation_label)
        res = dict()
        if relation_label is not None:
            loss = outputs[0]
            res['loss'] = loss
            logits = outputs[1]
            res['logits'] = logits
        else:
            logits = outputs[0]
            res['logits'] = logits
        return res

    def get_sim_score(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs[0]
        sim_scores = torch.softmax(logits, 1).data.tolist()
        return [x[1] for x in sim_scores]

    def get_nl_tokenizer(self):
        return self.tokenizer

    def get_pl_tokenizer(self):
        return self.tokenizer
