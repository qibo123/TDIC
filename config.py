import data_utils.const as const_util
class Config:
    def __init__(self):
        self.num_users = 100
        self.num_items = 200
        self.embedding_size = 64
        self.dis_loss = 'L2'
        self.dis_pen = 0.1
        self.int_weight = 0.5
        self.pop_weight = 0.3
        self.tide_weight = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 10
        self.model_type = 'TDIC' 
        self.num_layers = 2
        self.dropout = 0.5
        self.neg_sample_rate = 4
        self.dataset = 'myket' #'mb1m'
        self.load_path= const_util.myket
        self.margin = 40
        self.pool = 40
        self.shuffle = True
        self.num_workers = 8
        self.topk = [50]
        self.metrics = ['recall', 'ndcg']
        self.val_metrics = ['recall',  'ndcg']
        self.cg_use_gpu = True
        self.cg_gpu_id = 1
        self.checkpoint_dir='./check/'


