# tester.py

import torch
import numpy as np
from tqdm import tqdm
from absl import logging
 # 导入 config 文件
from metrics import Judger  # 假设你已经定义了 Judger 类
import faiss

class FaissInnerProductMaximumSearchGenerator:
    def __init__(self, flags_obj, items):
        self.items = items
        self.embedding_size = items.shape[1]
        self.make_index(flags_obj)

    def make_index(self, flags_obj):
        self.make_index_brute_force(flags_obj)

        if flags_obj.cg_use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, flags_obj.cg_gpu_id, self.index)

    def make_index_brute_force(self, flags_obj):
        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.index.add(self.items)

    def generate(self, users, k):
        _, I = self.index.search(users, k)
        return I

    def generate_with_distance(self, users, k):
        D, I = self.index.search(users, k)
        return D, I

class Tester:
    def __init__(self, config, model, dm,device):
        self.config = config
        self.model = model
        self.dm = dm
        self.judger = Judger(config, dm, max(config.topk))
        self.device = device
        self.max_topk = max(config.topk)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_items = []
        all_test_pos = []
        all_num_test_pos = []

        self.make_cg()

        with torch.no_grad():
            for batch_count, data in enumerate(tqdm(dataloader)):

                users, train_pos, test_pos, num_test_pos = data
                users = users.squeeze()



                items = self.cg(users, max(self.config.topk))
                items = self.filter_history(items, train_pos)

                all_items.append(items)
                all_test_pos.append(test_pos)
                all_num_test_pos.append(num_test_pos)

        all_items = torch.cat(all_items, dim=0)
        all_test_pos = torch.cat(all_test_pos, dim=0)
        all_num_test_pos = torch.cat(all_num_test_pos, dim=0)

        results, valid_num_users = self.judger.judge(all_items, all_test_pos, all_num_test_pos)

        for metric, value in results.items():
            results[metric] = value / valid_num_users

        return total_loss / len(dataloader), results

    def make_cg(self):
        self.item_embeddings = self.model.get_item_embeddings()
        self.generator = FaissInnerProductMaximumSearchGenerator(self.config, self.item_embeddings)
        self.user_embeddings = self.model.get_user_embeddings()

    def cg(self, users, topk):
        return self.generator.generate(self.user_embeddings[users], topk)

    def filter_history(self, items, train_pos):

        return np.stack([items[i][np.isin(items[i], train_pos[i], invert=True)][:self.max_topk] for i in range(len(items))], axis=0)