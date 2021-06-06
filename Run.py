import pickle, torch, os, time, random, sys
import numpy as np
import torch.optim as optim
import torch.nn as nn
from pytorch_pretrained_bert.optimization import BertAdam
from Modules import BertEncoder, SLModel
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from Config import Config
sys.path.append('./Utils')
from PrepareData import DataLoader, PrintMsg, Transform2Label, mkdir
from Evaluation import Performance

config = Config()
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)
os.environ['PYTHONHASHSEED'] = str(config.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

for split_inx in range(1, 11):

    data_path = config.datasplit_path
    save_path = config.save_path.format('model_name', split_inx)
    train, valid, test, _ = DataLoader(None, 'old', data_path, None, split_inx)
    train_len = len(train[0])
    train_iter_len = (train_len // config.batch_size) + 1

    base_encoder = BertEncoder(config)
    base_encoder.cuda()
    sl_model = SLModel(config)
    sl_model.cuda()
    nllloss = nn.NLLLoss()
    base_optimizer = filter(lambda x: x.requires_grad, list(base_encoder.parameters()))
    sl_optimizer = filter(lambda x: x.requires_grad, list(sl_model.parameters()))
    opt_params = [
            {'params': [p for p in sl_optimizer if len(p.data.size()) > 1], 'weight_decay': config.weight_decay},
            {'params': [p for p in sl_optimizer if len(p.data.size()) == 1], 'weight_decay': 0.0},
            {'params': base_optimizer, 'lr': config.base_lr},
            {'params': sl_optimizer}]
    opt = BertAdam(opt_params, lr=config.finetune_lr, warmup=config.warm_up, t_total=train_iter_len * config.epochs)
    
    total_batch, early_stop = 0, 0
    best_batch, best_f1 = 0, 0.0
    for epoch_i in range(config.epochs):
        batch_i = 0
        while batch_i * config.batch_size < train_len:
            base_encoder.train()
            sl_model.train()
            opt.zero_grad()
            
            start, end = batch_i * config.batch_size, (batch_i +1) * config.batch_size
            document_list = train[0][start: end]
            doc_len_list = [len(x.split('\x01')) for x in document_list]
            tag_labels, emo_labels, cau_labels = Transform2Label(train[1][start:end], doc_len_list, config)
            tag_labels_tensor = torch.LongTensor(tag_labels).cuda()
            emo_labels_tensor = torch.LongTensor(emo_labels).cuda()
            cau_labels_tensor = torch.LongTensor(cau_labels).cuda()
            
            _, clause_state_list = base_encoder(document_list)
            tag_log_probs, emo_log_probs, cau_log_probs = sl_model(clause_state_list, tag_labels, 'train' )
            tag_loss = nllloss(tag_log_probs, tag_labels_tensor)
            emo_loss = nllloss(emo_log_probs, emo_labels_tensor)
            cau_loss = nllloss(cau_log_probs, cau_labels_tensor)     

            loss = 0.50*tag_loss + 0.25*emo_loss + 0.25*cau_loss
            loss.backward()
            opt.step()
            batch_i += 1
            total_batch += 1

            if total_batch % config.showtime == 0:
                t_start = time.time()
                valid_emo, valid_cau, valid_pair = Performance(base_encoder, sl_model, valid, config)
                t_end = time.time()
                if valid_pair[2] > best_f1:
                    early_stop = 0
                    best_f1 = valid_pair[2]
                    best_batch = total_batch                    
                    print ('*'*50 +'the performance in valid set...' + '*'*50)
                    print('valid runging time: ', (t_end - t_start))
                    PrintMsg(total_batch, valid_emo, valid_cau, valid_pair) 
                    mkdir(save_path)
                    torch.save(base_encoder.state_dict(), save_path + '/bert_best.mdl_' + str(config.seed))
                    torch.save(sl_model.state_dict(), save_path + '/sl_best.mdl_' + str(config.seed))
        early_stop += 1
        if early_stop >= config.early_num or epoch_i == config.epochs-1:
            base_encoder.load_state_dict(torch.load(save_path + '/bert_best.mdl_' + str(config.seed)))
            sl_model.load_state_dict(torch.load(save_path + '/sl_best.mdl_' + str(config.seed)))
            print ('='*50 +'the performance in test set...' + '='*50)
            test_emo, test_cau, test_pair = Performance(base_encoder, sl_model, test, config)
            PrintMsg(best_batch, test_emo, test_cau, test_pair)
            pre, rec, f1 = test_pair[0], test_pair[1], test_pair[2]
            base_encoder_name = '/bertmodel_pre_' + str(pre) + '_rec_' + str(rec) + '_f1_' + str(f1)
            sl_name = '/slmodel_pre_' + str(pre) + '_rec_' + str(rec) + '_f1_' + str(f1)
            torch.save(base_encoder.state_dict(), save_path + base_encoder_name + '.mdl')
            torch.save(sl_model.state_dict(), save_path + sl_name + '.mdl')
            os.remove(save_path + '/bert_best.mdl_' + str(config.seed))
            os.remove(save_path + '/sl_best.mdl_' + str(config.seed))
            break