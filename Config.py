class Config(object):

    def __init__(self):
        
# >>>>>>>>>>>>>>>>>>>> For path <<<<<<<<<<<<<<<<<<<< #
        self.datasplit_path = './10-fold-data-splits'
        self.save_path = './10-fold-saved-models/{}/split_{:d}'

        self.bert_path = './bert-base-chinese'
# >>>>>>>>>>>>>>>>>>>> For training <<<<<<<<<<<<<<<<<<<< #        
        self.seed = 1024 #2000, 1024
        self.batch_size = 3 
        self.epochs = 10
        self.showtime = 100 
        self.base_lr = 1e-5 
        self.finetune_lr = 1e-3
        self.warm_up = 5e-2
        self.weight_decay = 1e-5
        self.early_num = 3
# >>>>>>>>>>>>>>>>>>>> For model <<<<<<<<<<<<<<<<<<<< #   
        self.is_bi = True

        self.gamma = 0.5
        self.cell_size = 256  
        self.layers = 1      
        self.bert_output_size = 768 
        self.mlp_size = 256   
        self.scale_factor = 2
        self.dropout = 0.5
        self.scope = 3 #3
        self.tag_ebd_dim = 128  # 128
