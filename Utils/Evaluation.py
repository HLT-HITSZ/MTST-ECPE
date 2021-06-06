# encoding: utf-8
# @author: ChuangFan
# email: fanchuanghit@gmail.com

from Metrics import emotion_metric, cause_metric, pair_metric
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
        
def ExtractPairs(tag_pred, len_list, config):
    start, scope = 0, config.scope
    ext_pairs = []
    for dl in len_list:
        end = start + dl
        doc_tag = tag_pred[start: end]
        pair = []
        for inx, tag in enumerate(doc_tag):
            if -scope <= tag-scope <= scope:
                pair.append((inx+tag-scope, inx))
        ext_pairs.append(pair)
        start = end
#     print (tag_pred)
    return ext_pairs

def pair2e_c_label(documents_len, data):
    emo_grounds, cau_grounds = [], []
    pair_list = data[1]
    for i, dl in enumerate(documents_len):
        pair = pair_list[i]
        emotion = [0] * dl
        cause = [0] * dl
        for p in pair:
            emotion[p[0]] = 1
            cause[p[1]] = 1
        emo_grounds.extend(emotion)
        cau_grounds.extend(cause)
    
    return emo_grounds, cau_grounds

def Performance(base_encoder, sl_model, data, config):   
    data_len = len(data[0])
    documents_len = [len(x.split('\x01')) for x in data[0]]
    emo_grounds, cau_grounds = pair2e_c_label(documents_len, data)
    pair_grounds = data[1]
    batch_i = 0
    base_encoder.eval()
    sl_model.eval()
    emo_preds, cau_preds, pair_preds = [], [], []
    while batch_i * config.batch_size < data_len:
        start, end = batch_i * config.batch_size, (batch_i +1) * config.batch_size
        document_list = data[0][start: end]
        pair_true_list = pair_grounds[start: end]
        len_list = documents_len[start: end]
        _, clause_state_list = base_encoder(document_list)
        retag_probs, emo_probs, cau_probs = sl_model(clause_state_list, None, 'eval')
        tag_pred = retag_probs.argmax(1).data.cpu().numpy().tolist()
        emo_pred = emo_probs.argmax(1).data.cpu().numpy().tolist()
        cau_pred = cau_probs.argmax(1).data.cpu().numpy().tolist()
        pair_pred = ExtractPairs(tag_pred, len_list, config)
        
        emo_preds.extend(emo_pred)
        cau_preds.extend(cau_pred)
        pair_preds.extend(pair_pred)
        batch_i += 1
    
    emo_metric = emotion_metric(pair_preds, pair_grounds)
    cau_metric = cause_metric(pair_preds, pair_grounds)
    pr_metric = pair_metric(pair_preds, pair_grounds)
    
    return (emo_metric, cau_metric, pr_metric)