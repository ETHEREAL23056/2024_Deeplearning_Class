import argparse

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from model import TransformerForClassification, device, TextClsDataset, TextGenDataset, TransformerForGeneration
from transformers import AutoTokenizer
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--train_cls', action='store_true', help='train classify model')
    parser.add_argument('--train_gen', action='store_true', help='train generate model')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--head_number', type=int, default=1, help='head number')
    parser.add_argument('--layer_number', type=int, default=6, help='layer number')
    parser.add_argument('--class_number', type=int, default=40, help='class number')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_cls:
        # 导入训练数据
        train_dataset = pd.read_csv('./dataset/train_data.csv')

        # 去除空值
        train_dataset = train_dataset[train_dataset['short_description'].notna()].reset_index(drop=True)
        print(train_dataset.shape)

        # 数据量太大了，单卡情况下的模型训练需要的时间太长，考虑适当降低数据量进行训练
        train_dataset = train_dataset.sample(frac=.01, random_state=42).reset_index(drop=True)
        print(train_dataset.shape)

        # 训练模型
        LR = args.lr
        EPOCHS = args.epoch
        BATCH_SIZE = 16
        MAX_LENGTH = 512

        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_data = TextClsDataset(train_dataset['short_description'], train_dataset['categoryEncoded'], tokenizer,
                                    MAX_LENGTH)
        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        VOCAB_SIZE = tokenizer.vocab_size
        D_MODEL = 512
        NUM_HEADS = args.head_number
        NUM_LAYERS = args.layer_number
        D_FF = 1024
        MAX_SEQ_LENGTH = 512
        DROPOUT = .1
        NUM_CLASSES = 40

        model = TransformerForClassification(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LENGTH,
                                             NUM_CLASSES,
                                             DROPOUT, device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_list = []

        for epoch in range(EPOCHS):
            loss_sum = 0
            for batch in train_data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs, _ = model(input_ids)
                loss = criterion(outputs, labels)
                loss_sum += loss.item()

                loss.backward()
                optimizer.step()

            batch_avg_loss = loss_sum / len(train_data_loader)
            loss_list.append(batch_avg_loss)
            print(f"loss of epoch {epoch} per batch: {batch_avg_loss}; per sample: {batch_avg_loss / BATCH_SIZE}")

        # 绘制损失曲线
        plt.plot(loss_list, label="Loss")
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.show()

        # 保存模型
        torch.save(model.state_dict(), f'./model/cls_model_{LR}_{EPOCHS}_{NUM_HEADS}_{NUM_LAYERS}.pth')
    else:
        # 导入训练数据
        train_dataset = pd.read_csv('./dataset/train_data_gen.csv')

        # 去除空值
        train_dataset = train_dataset[train_dataset['short_description'].notna()].reset_index(drop=True)
        print(train_dataset.shape)

        # 数据量太大了，单卡情况下的模型训练需要的时间太长，考虑适当降低数据量进行训练
        train_dataset = train_dataset.sample(frac=.01, random_state=42).reset_index(drop=True)
        print(train_dataset.shape)

        # 训练模型
        LR = args.lr
        EPOCHS = args.epoch
        BATCH_SIZE = 16
        MAX_LENGTH = 512

        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_data = TextGenDataset(train_dataset['short_description'], train_dataset['headline'], tokenizer,
                                    tokenizer, MAX_LENGTH)
        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = tokenizer.vocab_size, tokenizer.vocab_size
        D_MODEL = 512
        NUM_HEADS = args.head_number
        NUM_LAYERS = args.layer_number
        D_FF = 1024
        MAX_SEQ_LENGTH = 512
        DROPOUT = .1

        model = TransformerForGeneration(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF,
                                         MAX_SEQ_LENGTH, DROPOUT, device)
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_list = []
        for epoch in range(EPOCHS):
            loss_sum = 0
            for batch_idx, (src, tgt) in enumerate(train_data_loader):
                src = src.to(device)
                tgt = tgt.to(device)
                src_mask, tgt_mask = model.generate_mask(src, tgt)

                optimizer.zero_grad()
                logits, _ = model(src, tgt[:, :-1])
                loss = criterion(logits.view(-1, logits.size(-1)), tgt[:, 1:].contiguous().view(-1))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            batch_avg_loss = loss_sum / len(train_data_loader)
            loss_list.append(batch_avg_loss)
            print(f"loss of epoch {epoch} per batch: {batch_avg_loss}; per sample: {batch_avg_loss / BATCH_SIZE}")

        # 绘制损失曲线
        plt.plot(loss_list, label="Loss")
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.show()

        # 保存模型
        torch.save(model.state_dict(), f'./model/gen_model_{LR}_{EPOCHS}_{NUM_HEADS}_{NUM_LAYERS}.pth')


main()
