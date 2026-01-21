import tqdm
import numpy as np
import torch
import os
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random

class RecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []  # 存放每个用户交互序列去掉answer后的序列的所有前缀序列
        self.max_len = args.max_seq_length
        self.user_ids = []
        self.contrastive_learning = args.model_type.lower() in ['fearec', 'duorec'] # 是否运用对比学习
        self.data_type = data_type  # 数据状态(训练、测试、验证)

        # 构造训练集
        if self.data_type=='train':
            # enumerate同时取出元素和索引
            # user_seq: (N, L)     N: 用户数     L: 用户交互的物品
            for user, seq in enumerate(user_seq):

                # 去除序列最后两个元素以后，input_ids取序列的最后的max_len个元素。max_len表示模型考虑用户最近最大交互的物品数，如果max_len超出seq长度，也会从seq的0索引取起
                # 为什么要去掉最后2个元素而不是1个? -> seq[-2]（倒数第二个物品）可能与 seq[-1] 强相关  (GPT)
                # 我觉得还有可能是留1个位置给验证集? 因为验证集也是取最后一个元素
                input_ids = seq[-(self.max_len + 2):-2]   # [1, 2, 3]

                # 拆解成这种前缀子序列是为了丰富数据吗? -> 是为了让模型学习从短到长的历史序列，预测未来的交互物品。如果用完整序列，无法学习从短到长的兴趣演变
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])   # [[1], [1, 2], [1, 2, 3]]
                    self.user_ids.append(user)                # [[0], [0], [0]]

        # valid（验证集）：去掉用户交互序列的最后一个物品，用于验证模型是否能正确预测最后一个物品
        elif self.data_type=='valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])

        # test（测试集）：直接使用完整序列，测试模型的最终性能
        # 测试集需要有完整的序列，在测试步骤的时候，__getitem__分别取出input和answer
        else:
            self.user_seq = user_seq

        self.test_neg_items = test_neg_items

        if self.contrastive_learning and self.data_type=='train':
            if os.path.exists(args.same_target_path):
                self.same_target_index = np.load(args.same_target_path, allow_pickle=True)
            else:
                print("Start making same_target_index for contrastive learning")
                self.same_target_index = self.get_same_target_index()
                self.same_target_index = np.array(self.same_target_index)
                np.save(args.same_target_path, self.same_target_index)

    def get_same_target_index(self):
        num_items = max([max(v) for v in self.user_seq]) + 2
        same_target_index = [[] for _ in range(num_items)]
        
        user_seq = self.user_seq[:]
        tmp_user_seq = []
        for i in tqdm.tqdm(range(1, num_items)):
            for j in range(len(user_seq)):
                if user_seq[j][-1] == i:
                    same_target_index[i].append(user_seq[j])
                else:
                    tmp_user_seq.append(user_seq[j])
            user_seq = tmp_user_seq
            tmp_user_seq = []

        return same_target_index

    def __len__(self):
        return len(self.user_seq)

    # __getitem__ 方法是Pytorch的Dataset类的标准方法，RecDataset继承了Dataset类
    def __getitem__(self, index):

        # 为了能够用dataset[i]的索引方式，现在要获取输入(input_ids)和预测(answer)
        items = self.user_seq[index] # 取出测试用户的完整交互序列
        input_ids = items[:-1]   # 取前 n-1 个物品作为输入
        answer = items[-1]   # 真实的下一个交互物品


        # 负样本 -> 模型不仅要预测正确答案 answer，还要能分辨出错误答案 neg_answer（负样本）
        seq_set = set(items)    # 用户历史交互物品
        neg_answer = neg_sample(seq_set, self.args.item_size)   # 负样本，模型应该预测它的概率低


        # padding（填充 0 让 input_ids 长度一致，具体的长度由args中的 max_seq_length 定义）
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids   # 在前面填充 0
        input_ids = input_ids[-self.max_len:]   # 确保长度为 max_len
        assert len(input_ids) == self.max_len   # 确保长度一致

        # 1. valid or test -> 不需要负样本
        if self.data_type in ['valid', 'test']:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
                torch.zeros(0, dtype=torch.long), # not used
            )

        # 2. contrastive_learning（对比学习） → 需要 sem_aug（语义增强样本）
        elif self.contrastive_learning:
            sem_augs = self.same_target_index[answer]
            sem_aug = random.choice(sem_augs)
            keep_random = False
            for i in range(len(sem_augs)):
                if sem_augs[0] != sem_augs[i]:
                    keep_random = True

            while keep_random and sem_aug == items:
                sem_aug = random.choice(sem_augs)

            sem_aug = sem_aug[:-1]
            pad_len = self.max_len - len(sem_aug)
            sem_aug = [0] * pad_len + sem_aug
            sem_aug = sem_aug[-self.max_len:]
            assert len(sem_aug) == self.max_len

            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(sem_aug, dtype=torch.long)
            )

        # 3. train（普通训练） → 需要负样本 neg_answer
        else:
            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
            )

        return cur_tensors


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):

        for item in item_list[:-2]:
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    '''
    构造了一个稀疏矩阵 (csr_matrix)，它是 scipy.sparse 提供的一种 高效存储稀疏数据的方式，数据格式的样例如下
    [[1 0 1 0]  # 用户 0 交互了物品 0, 2
    [0 0 1 0]  # 用户 1 交互了物品 2
    [1 1 1 0]] # 用户 2 交互了物品 0, 1, 2
    稀疏矩阵只存储非零元素，计算时跳过零值，提高计算速度
    '''
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        # test评分矩阵的构造只去掉最后一个物品，目的是让 test 预测最后的物品，评估最终推荐效果
        for item in item_list[:-1]:
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):
    
    num_items = max_item + 1

    # 评分矩阵 -> 不含answer的交互记录矩阵
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []        # 二维列表，其中下标为 i 的元素列表表示id为 i + 1 的用户的物品交互列表
    item_set = set()     # 一维列表，存储用户交互过的物品id
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')   # 生成item列表
        items = [int(item) for item in items]   # 把item里的物品id变为int类型
        user_seq.append(items)
        item_set = item_set | set(items)        # |是求并集，用于收集用户交互过的物品的集合
    max_item = max(item_set)   # 用户交互过的物品的最大id
    num_users = len(lines)     # 用户id的最大值

    return user_seq, max_item, num_users

def get_seq_dic(args):

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, num_users = get_user_seqs(args.data_file)
    seq_dic = {'user_seq':user_seq, 'num_users':num_users }

    return seq_dic, max_item, num_users

def get_dataloder(args,seq_dic):

    train_dataset = RecDataset(args, seq_dic['user_seq'], data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    eval_dataset = RecDataset(args, seq_dic['user_seq'], data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    test_dataset = RecDataset(args, seq_dic['user_seq'], data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_dataloader, eval_dataloader, test_dataloader
