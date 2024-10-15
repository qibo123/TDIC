# trainer.py

import torch
from model import TDIC, LGNTDIC
import torch.optim as optim
from tqdm import tqdm
from absl import logging
import time
import numpy as np
from tester import Tester
from dataloader import CGDataProcessor

class Trainer:
    def __init__(self, config, device, dm):
        self.dm = dm
        self.config = config
        self.device = device
        self.model = self._create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.tester = Tester(config, self.model, dm,self.device)
        self.best_metrics = None
        self.val_dataloader, self.topk_margin = CGDataProcessor.get_dataloader(self.config, 'val')

    def _create_model(self):
        if self.config.model_type == 'TDIC':
            return TDIC(
                self.dm.num_users,
                self.dm.num_items,
                self.config.embedding_size,
                self.config.dis_loss,
                self.config.dis_pen,
                self.config.int_weight,
                self.config.pop_weight,
                self.config.tide_weight
            ).to(self.device)
        elif self.config.model_type == 'LGNTDIC':
            return LGNTDIC(
                self.config.num_users,
                self.config.num_items,
                self.config.embedding_size,
                self.config.num_layers,
                self.config.dropout,
                self.config.dis_loss,
                self.config.dis_pen,
                self.config.int_weight,
                self.config.pop_weight,
                self.config.tide_weight
            ).to(self.device)
        else:
            raise ValueError("Invalid model type")

    def train(self, train_dataloader):
        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(train_dataloader)
            val_loss, metrics = self.tester.evaluate(self.val_dataloader)

            # 打印每个 epoch 的结果
            print(f"Epoch {epoch+1}/{self.config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            for metric, value in metrics.items():
                print(f"Validation {metric}: {value:.4f}")

            # 更新最好的评估指标
            if self.best_metrics is None or metrics['recall'] > self.best_metrics['recall']:
                self.best_metrics = metrics
                self.save_model(epoch)

        # 打印最好的评估指标
        print("\nBest Validation Metrics:")
        for metric, value in self.best_metrics.items():
            print(f"Best {metric}: {value:.4f}")

    def _train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        # 使用 tqdm 包装 dataloader
        for batch_count, sample in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
            user, item_p, item_n, mask = sample
            user, item_p, item_n, mask = user.to(self.device), item_p.to(self.device), item_n.to(self.device), mask.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model(user, item_p, item_n, mask)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def save_model(self, epoch):
        save_path = f"{self.config.checkpoint_dir}/model_epoch_{epoch}.pth"
        torch.save(self.model.state_dict(), save_path)
        logging.info(f"Model saved at {save_path}")