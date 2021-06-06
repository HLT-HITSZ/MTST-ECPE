import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import pickle
from pytorch_pretrained_bert import BertModel, BertTokenizer

class BertEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    
    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list
        
    def forward(self, document_list):
        text_list, tokens_list, ids_list = [], [], []
        ## The clauses in each document are splited by '\x01'
        document_len = [len(x.split('\x01')) for x in document_list]
        
        for document in document_list:
            text_list.extend(document.strip().split('\x01'))  
        for text in text_list:
            text = ''.join(text.split())
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            tokens_list.append(tokens)
        for tokens in tokens_list:
            ids_list.append(self.tokenizer.convert_tokens_to_ids(tokens))
                
        ids_padding_list, mask_list = self.padding_and_mask(ids_list)
        ids_padding_tensor = torch.LongTensor(ids_padding_list).cuda()
        mask_tensor = torch.tensor(mask_list).cuda()
        
        _, pooled = self.bert(ids_padding_tensor, attention_mask = mask_tensor, output_all_encoded_layers=False)
        
        start = 0
        clause_state_list = []
        for dl in document_len:
            end = start + dl
            clause_state_list.append(pooled[start: end])
            start = end
        return pooled, clause_state_list
    
    
class SLModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.is_bi = config.is_bi
        self.bert_output_size = config.bert_output_size
        self.mlp_size = config.mlp_size
        self.cell_size = config.cell_size
        self.scale_factor = config.scale_factor
        self.dropout = config.dropout
        self.layers = config.layers
        self.gamma = config.gamma
        self.scope = config.scope
        self.tag_ebd_dim = config.tag_ebd_dim
        self.tag_embedding = nn.Embedding(self.scope*2+1+1+1+1, self.tag_ebd_dim)

        self.encoder = nn.LSTM(self.bert_output_size, self.cell_size, self.layers, bidirectional=self.is_bi)
        self.decoder = nn.LSTM(self.cell_size*2 + self.tag_ebd_dim if self.is_bi else self.cell_size + self.tag_ebd_dim,
                               self.cell_size, self.layers, bidirectional=False)

        self.tag_MLP = nn.Sequential(
            nn.Linear(self.cell_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, self.scope*2+1+1)
         )
        self.emo_MLP = nn.Sequential(
            nn.Linear(self.cell_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, 2)
         )
        self.cau_MLP = nn.Sequential(
            nn.Linear(self.cell_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, 2)
         )

        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                if len(param.data.size()) > 1:
                    nn.init.xavier_normal(param.data)
                else:
                    param.data.uniform_(-0.1, 0.1)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue
        self.tag_embedding.weight.requires_grad = False

    def init_hidden(self, batch_size, mode):

        if self.is_bi and mode == 'encoder':
            hidden = [Variable(torch.zeros(self.layers*2, batch_size, self.cell_size).cuda()),
                      Variable(torch.zeros(self.layers*2, batch_size, self.cell_size).cuda())
                     ]
        else:
            hidden = [Variable(torch.zeros(self.layers, batch_size, self.cell_size).cuda()),
                      Variable(torch.zeros(self.layers, batch_size, self.cell_size).cuda())
                     ]
        return hidden

    def Encoder(self, clause_state_list, len_list):
        state_dim = clause_state_list[0].size()[1]
        bs = len(clause_state_list)
        max_len = max(len_list)
        clause_padding_list = []
        for x in clause_state_list:
            clause_padding = torch.cat([x, Variable(torch.zeros(max_len-x.size()[0],state_dim).cuda())], 0)
            clause_padding_list.append(clause_padding)
        en_inputs = torch.stack(clause_padding_list, 0).permute(1, 0, 2)
        init_hidden = self.init_hidden(bs, 'encoder')
        en_outputs, _ = self.encoder(en_inputs, init_hidden)
        en_outputs = en_outputs.permute(1, 0, 2)

        return en_outputs

    def Decoder(self, en_outputs, len_list, tag_labels):
        start, bs = 0, len(len_list)
        tag_inputs = []
        max_len = max(len_list)
        for dl in len_list:
            end = start + dl
            tags = tag_labels[start: end]
            temp = [self.scope*2+1+1] + tags[:-1] + [self.scope*2+1+1+1]*(max_len-dl)
            tag_inputs.append(torch.LongTensor(temp).cuda())
            start = end
        tag_inputs_tensor = torch.stack(tag_inputs, 0)
        tag_inputs_ebd = self.tag_embedding(tag_inputs_tensor)
        de_inputs = torch.cat([en_outputs, tag_inputs_ebd], 2).permute(1, 0, 2)
        init_state = self.init_hidden(bs, 'decoder')
        de_outputs, _ = self.decoder(de_inputs, init_state)
        de_outputs = de_outputs.permute(1, 0, 2)

        return de_outputs

    def ProbsBias(self, i, tag_probs_seg, cau_probs_seg, e_i):

        num_seg = tag_probs_seg.size()[0]
        num_pre = self.scope - i if self.scope - i > 0 else 0
        top_padding = torch.zeros(num_pre, self.scope*2+1+1).cuda()
        cau_probs_seg = torch.cat([torch.FloatTensor([0]*num_pre).cuda(), cau_probs_seg])
        num_tail = self.scope - (num_seg-i-1) if self.scope - (num_seg-i-1) > 0 else 0
        tail_padding = torch.zeros(num_tail, self.scope*2+1+1).cuda()
        cau_probs_seg = torch.cat([cau_probs_seg, torch.FloatTensor([0]*num_tail).cuda()])
        padding_seg = torch.cat([top_padding, tag_probs_seg, tail_padding], 0)

        bias = []
        for i in range (self.scope*2+1):
            total = 1 - padding_seg[i][self.scope*2-i]
            p_weight = 1 - (abs(i-self.scope)+self.gamma)/(self.scope+self.gamma*2)
            e_weight = e_i
            c_weight = cau_probs_seg[i]
            if e_i > 0.5:
                v = c_weight * e_weight * p_weight * total
            else:
                v = (1-c_weight) * (1-e_weight) * (1-p_weight) * (1-total)
            temp = [-v/(self.scope*2+1) if _ != self.scope*2-i else v for _ in range(self.scope*2+1+1)]
            bias.append(torch.FloatTensor(temp).cuda())
        bias = bias[num_pre: num_pre+num_seg]
        bias = torch.stack(bias, 0)

        return bias


    def ProbsRevised(self, tag_probs_i, emo_i, cau_i):

        for i, e_i in enumerate(emo_i):
            len_i = tag_probs_i.size()[0]
            start = 0 if i - self.scope < 0 else i - self.scope
            end = -1 if i + self.scope > len_i - 1 else i + self.scope + 1
            tag_probs_seg = tag_probs_i[start: end]
            cau_probs_seg = cau_i[start: end]
            bias = self.ProbsBias(i, tag_probs_seg, cau_probs_seg, e_i)

            if e_i < 0.5:
                tag_probs_i[start: end] = tag_probs_i[start: end] - bias
            else:
                tag_probs_i[start: end] = tag_probs_i[start: end] + bias

        return tag_probs_i


    def ProbsDrifter(self, tag_probs, emo_probs, cau_probs, len_list):
        retag_probs_list = []
        start = 0
        for l in len_list:
            end = start + l
            tag_probs_i = tag_probs[start: end]
            emo_probs_i = emo_probs[start: end]
            cau_probs_i = cau_probs[start: end]
            emo_i, cau_i = emo_probs_i[:, 1], cau_probs_i[:, 1]
            retag_probs_i = self.ProbsRevised(tag_probs_i, emo_i, cau_i)
            start = end
            retag_probs_list.append(retag_probs_i)

        retag_probs = torch.cat(retag_probs_list, 0)
        return retag_probs


    def Eval(self, en_outputs, len_list):
        bs = len(len_list)
        inputs = en_outputs.permute(1, 0, 2)
        tag_inputs = [self.scope*2+1+1] * bs
        tag_inputs_tensor = torch.LongTensor(tag_inputs).cuda()
        tags_inputs_ebd = self.tag_embedding(tag_inputs_tensor).unsqueeze(0)
        init_state = self.init_hidden(bs, 'decoder')
        feature_list = []
        for step_i in inputs:
            step_i = step_i.unsqueeze(0)
            inputs_i = torch.cat([step_i, tags_inputs_ebd], 2)
            output, init_state = self.decoder(inputs_i, init_state)
            feature_list.append(output.squeeze(0))
            tag_inputs = self.tag_MLP(output.squeeze(0)).argmax(1).data.cpu().numpy().tolist()
            tag_inputs_tensor = torch.LongTensor(tag_inputs).cuda()
            tags_inputs_ebd = self.tag_embedding(tag_inputs_tensor).unsqueeze(0)
        features = torch.stack(feature_list, 0).permute(1, 0, 2)
        features = torch.cat([features[i][:l] for i, l in enumerate(len_list)], 0)
        tag_logits = self.tag_MLP(features)
        emo_logits = self.emo_MLP(features)
        cau_logits = self.cau_MLP(features)

        tag_probs = F.softmax(tag_logits, 1)
        emo_probs = F.softmax(emo_logits, 1)
        cau_probs = F.softmax(cau_logits, 1)
        retag_probs = self.ProbsDrifter(tag_probs, emo_probs, cau_probs, len_list)

        return retag_probs, emo_probs, cau_probs

    def forward(self, clause_state_list, tag_labels, mode):
        len_list = [x.size()[0] for x in clause_state_list]
        if mode == 'train':
            en_outputs = self.Encoder(clause_state_list, len_list)
            de_outputs = self.Decoder(en_outputs, len_list, tag_labels)
            de_outputs_list = []
            for i, l in enumerate(len_list):
                de_outputs_list.append(de_outputs[i][:l])

            features = torch.cat(de_outputs_list, 0)
            tag_logits = self.tag_MLP(features)
            emo_logits = self.emo_MLP(features)
            cau_logits = self.cau_MLP(features)

            tag_probs = F.softmax(tag_logits, 1)
            emo_probs = F.softmax(emo_logits, 1)
            cau_probs = F.softmax(cau_logits, 1)

            emo_log_probs = torch.log(emo_probs)
            cau_log_probs = torch.log(cau_probs)
            tag_log_probs = torch.log(tag_probs)
            return tag_log_probs, emo_log_probs, cau_log_probs
        else:
            en_outputs = self.Encoder(clause_state_list, len_list)
            retag_probs, emo_probs, cau_probs = self.Eval(en_outputs, len_list)

            return retag_probs, emo_probs, cau_probs