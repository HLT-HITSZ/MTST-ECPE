# encoding: utf-8
# @author: ChuangFan
# email: fanchuanghit@gmail.com

import pickle
import numpy as np
import os

def mkdir(path):
    import os
    path=path.strip()
    path=path.rstrip(r"/")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        print ('the directory already existes!')
        return False

    
def DataLoader(doc2pair, mode, save_path, config, inx):
    if mode == 'new':
        # the new splits are named by their created time, in released version, we rename them to split_1, split_2,...,split_20
        dt = datetime.datetime.now()
        path_name = dt.strftime('%Y-%m-%d--%H-%M-%S')
        save_path = config.datasplit_path + '/' + path_name
        mkdir(save_path)
        return DataSplit(doc2pair, save_path)
    else:
        # loading already exists data split
        train = pickle.load(open(save_path + '/train_{}.pkl'.format(inx), 'rb'))
        valid = pickle.load(open(save_path + '/valid_{}.pkl'.format(inx), 'rb'))
        test = pickle.load(open(save_path + '/test_{}.pkl'.format(inx), 'rb'))
        return train, valid, test, save_path
                
    
def DataSplit(doc2pair, save_path):
    length = len(doc2pair)
    split_1, split_2 = int(length * 0.8), int(length * 0.9) 
    data, labels = [], []
    for k in doc2pair:
        data.append(k)
        labels.append(doc2pair[k])
    inx = list(range(length))
    np.random.shuffle(inx)
    train_data, train_label = [], []
    valid_data, valid_label = [], []
    test_data, test_label = [], []
    for i, j in enumerate(inx):
        if i < split_1:
            train_data.append(data[j]), train_label.append(labels[j])
        elif split_1 <= i < split_2:
            valid_data.append(data[j]), valid_label.append(labels[j])
        else:
            test_data.append(data[j]), test_label.append(labels[j])
    train = [train_data, train_label]
    valid = [valid_data, valid_label]
    test = [test_data, test_label]
    
    pickle.dump(train, open(save_path + '/train.pkl', 'wb'))
    pickle.dump(valid, open(save_path + '/valid.pkl', 'wb'))
    pickle.dump(test, open(save_path + '/test.pkl', 'wb'))
    
    return train, valid, test, save_path
    
def Transform2Label(pair_list, doc_len_list, config):
    scope = config.scope
    emo_labels, cau_labels, tag_labels = [], [], []
    for i, dl in enumerate(doc_len_list):
        emotions = [0]*dl
        causes = [0]*dl
        temp = [(x[0], x[1]) for x in pair_list[i]]  # delete future
        pairs = set(temp)
        for pr in pairs:
            emotions[pr[0]] = 1
            causes[pr[1]] = 1
        emo_labels.extend(emotions)
        cau_labels.extend(causes)
        
        tags = []
        for c_id in range(dl):
            tl = scope*2 + 1
            tags.append(tl)
            for pr in pairs:
                if c_id == pr[1]:
                    tl = pr[0]-pr[1]+scope if -scope<= pr[0]-pr[1] <= scope else scope*2+1
                    tags[-1] = tl
                    break
#                 elif c_id == pr[0]:
#                     tl = scope*2 + 1
#                     tags[-1] = tl
#                     break
        tag_labels.extend(tags)
    return tag_labels, emo_labels, cau_labels
    
    
def PrintMsg(total_batch, emo_metric, cse_metric, pr_metric):
    emo_msg = 'total batch: {}, emo pre: {:.4f}, emo rec: {:.4f}, emo f1: {:.4f}'.format(total_batch, emo_metric[0], emo_metric[1], emo_metric[2])
    cse_msg = 'total batch: {}, cse pre: {:.4f}, cse rec: {:.4f}, cse f1: {:.4f}'.format(total_batch, cse_metric[0], cse_metric[1], cse_metric[2])
    pr_msg = 'total batch: {}, pr pre: {:.4f}, pr rec: {:.4f}, pr f1: {:.4f}'.format(total_batch, pr_metric[0], pr_metric[1], pr_metric[2])
    print (emo_msg + '\n' + cse_msg + '\n' + pr_msg) 