import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 假设每行是一个JSON对象，包含'text'字段
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 获取样本
        sample = self.samples[index]

        # 构建输入文本
        # 对文本进行tokenizer编码
        
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # 将输入ID张量化并去掉批次维度
        # input_ids shape: (seq_len,)
        input_ids = encoding.input_ids.squeeze()
        # 计算 loss mask，非pad位置为1，pad位置为0，pad 的地方不计算 loss
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        # 自回归任务，构建X和Y
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        # 对齐预测位置
        # 由于Y是input_ids右移一位，所以loss_mask也需要右移一位
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        # 返回数据
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 将特定的控制字符和文本转换为模型能够理解的整数列表
        # 这些ID用于识别对话的开始和结束
        # add_special_tokens=False 确保只获取指定文本的ID，而不添加额外的特殊标记
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        # 类似于 PretrainDataset 的 load_data 方法
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, cs):
        # 构建符合ChatML格式的对话
        # cs 代表 conversations 列表

        # 复制对话列表，避免修改原始数据
        messages = cs.copy()
        # tools 变量用于存储系统消息中的工具信息（如果有的话）
        # 如果第一个消息的 role 是 "system" 且包含"functions"字段，则提取该字段作为tools
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        # 构建符合ChatML格式的对话
        # 返回时不进行tokenize，并且不添加生成提示符
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _generate_loss_mask(self, input_ids):
        # 构建动态损失掩码
        # 只有assistant回复的部分才计算loss

        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 遍历查找assistant回复的开始位置
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 找到assistant回复的起始位置
                start = i + len(self.bos_id)
                end = start
                # 遍历查找assistant回复的结束位置
                while end < len(input_ids):
                    # 找到回复的结束位置
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 标记loss_mask中对应assistant回复的部分为1
                # 从start+1开始，因为第一个token是assistant的标识符，不计算loss
                # 标记到end加上eos_id的长度，确保包含结束标识符
                # 标记时确保不超过max_length
                # 注意标识符问题，开始标识符不加入 loss 是因为它只是一个标记，是模型输入的一部分，模型不需要生成这部分
                # 而结束标识符是模型必须学会生成的内容的一部分
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 更新i位置，继续查找下一个assistant回复
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        # 对提示进行tokenizer编码，确保长度不超过max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 如果长度不足max_length，则进行padding
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        # 只有assistant回复的部分才计算loss
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        # 输入是input_ids的前n-1个token
        # 目标是input_ids的第2到第n个token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        # 读取JSONL文件
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    pass
