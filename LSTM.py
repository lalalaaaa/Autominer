from torch.nn import functional as F
from const import *
import torch
import torch.nn as nn
import Load_input_test as data_load
import torch.sparse as sp
import datetime #系统时间作为文件名，防止记录时同文件名覆盖
import miner_4_LSTM_test as miner_test
import MCTS_4_Alpha_Miner as MCTS
import os
import random

window_size = INPUT_WINDOW_SIZE
max_num_decision = OUTPUT_BROADCAST_LEN
# EMBED_DIM = 16
# time step
max_move = LSTM_TIME_STEP
# EMBEDDING dim
num_embeddings = INPUT_FEATURE_DIM
embedding_dim = EMBED_DIM
# LSTM input feature dim
input_feature_dim = LSTM_INPUT_FEATURE_DIM # 4 chains multi-hot encode
# LSTM output feature dim
output_feature_dim = LSTM_OUTPUT_FEATURE_DIM# 2 INPUT_WINDOW_SIZE actions multi-hot encode
num_epochs = NUM_EPOCHS
num_loader = NUM_LOADER


class AlphaMinerModel(torch.nn.Module):
    def __init__(self):
        super(AlphaMinerModel, self).__init__()
        #self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = torch.nn.LSTM(input_feature_dim, output_feature_dim, NUM_HIDDEN, bias=True, dropout=0.3, batch_first=True)
        #self.output_hidden = torch.nn.Linear(output_feature_dim, output_feature_dim, bias=True)
        self.output_broadcast = torch.nn.Linear(output_feature_dim, MAX_BLOCK, bias=True)
        self.output_mine = torch.nn.Linear(output_feature_dim, MAX_BLOCK, bias=True)

    def get_param_num(self):
        num_lstm_params = sum(p.numel() for p in self.lstm.parameters())
        num_brd_params = sum(p.numel() for p in self.output_broadcast.parameters())
        num_mine_params = sum(p.numel() for p in self.output_mine.parameters())
        num_param = num_lstm_params+num_brd_params+num_mine_params
        return num_param

    @classmethod
    def load_model(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

    def forward(self, input):
        # Convert input to Long type
        #input = input.long()
        #embedded_input = self.embedding(input)
        # Reshape back to 3D
        #batch_size, time_step, feature_dim, embed_dim = embedded_input.size()
        #embedded_input = embedded_input.view(batch_size, time_step, feature_dim * embed_dim)

        # Initialize hidden state with ones
        # constant_value = torch.tensor(0.1)
        h0 = torch.zeros(NUM_HIDDEN, BATCH_SIZE, output_feature_dim).requires_grad_()
        c0 = torch.zeros(NUM_HIDDEN, BATCH_SIZE, output_feature_dim).requires_grad_()
        #for i in range(max_move):
        out, (hn, cn) = self.lstm(input, (h0.detach(), c0.detach()))
        #out = self.output_hidden(out) #add one hidden linear layer
        broadcast_decision = self.output_broadcast(out)
        broadcast_decision = torch.sigmoid(broadcast_decision)
        mine_decision = self.output_mine(out)
        mine_decision = F.softmax(mine_decision, dim=-1)

        return broadcast_decision, mine_decision

    def step_forward(self, input, curr_h=torch.zeros(NUM_HIDDEN, 1, output_feature_dim).requires_grad_(), curr_n=torch.zeros(NUM_HIDDEN, 1, output_feature_dim).requires_grad_()):
        with torch.no_grad():
            # Reshape the input to match LSTM's expected shape
            input = input.view(1, 1, -1)  # (1, 1, feature_dim)
            input = input.float()
            out, (curr_h, curr_n) = self.lstm(input, (curr_h.detach(), curr_n.detach()))
            #out = self.output_hidden(out)  # add one hidden linear layer
            broadcast_decision = self.output_broadcast(out)
            broadcast_decision = torch.sigmoid(broadcast_decision) #真正的广播序号化（int）会在test.py中进行处理，具体见multi_hot_to_broadcast函数。
            mine_decision = self.output_mine(out)
            mine_decision = F.softmax(mine_decision, dim=-1)

        return broadcast_decision, mine_decision, curr_h, curr_n

def train_model(model, diff_learn_rate):
    #error_broadcast = nn.BCELoss(weight=, reduction='mean') #二元交叉损失熵
    #error_mine = nn.KLDivLoss(reduction='batchmean')
    error_mine = nn.CrossEntropyLoss()
    error_mine = nn.L1Loss()
    #统计文件夹数量
    def count_files_in_folder(folder_path):
        file_count = 0
        for _, _, files in os.walk(folder_path):
            # 忽略子文件夹，只统计当前文件夹
            for file in files:
                file_count += 1
        return file_count

    if diff_learn_rate:
        for train_data_i in range(4):
            learning_rate = 0.0001 * train_data_i
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            print("////////////train_data: " + str(train_data_i+1) + "////////////")
            num_epoch = NUM_EPOCHS/(2**train_data_i)
            for epoch in range(int(num_epoch)):
                for loader in range(num_loader):
                    one_batch_train_data, one_batch_broadcast_label_data, one_batch_mine_label_data = data_load.load_one_batch_data(TRAIN_DATA_PATH + str(train_data_i+1),batch_round=loader)
                    optimizer.zero_grad()

                    # 对input进行0值的添加处理
                    value_to_add = one_batch_train_data.mean() #求得非0数值的平均值

                    # 对input中的1的左右的0值进行微增处理
                    increment = 0.1  # 定义微小增量的值
                    indices = one_batch_train_data.nonzero(as_tuple=False)  # 获取稀疏张量中值为1的元素的索引
                    for idx in indices:
                        plane, row, col = idx.tolist()
                        # 对左侧进行微小增量操作
                        if col > 0:
                            one_batch_train_data[plane][row][col - 1] += increment
                        # 对右侧进行微小增量操作
                        if col < one_batch_train_data.size(2) - 1:
                            one_batch_train_data[plane][row][col + 1] += increment

                    one_batch_train_data = torch.add(one_batch_train_data, value_to_add)

                    broadcast_decision, mine_decision = model(one_batch_train_data)

                    #计算广播的损失权重（使用Weighted Cross Entropy Loss）：
                    weight = one_batch_broadcast_label_data
                    # 统计one_batch_broadcast_label_data中0的个数
                    count_zero = torch.sum(torch.eq(weight, 0)).item()
                    # 统计one_batch_broadcast_label_data中1的个数
                    count_one = torch.sum(torch.eq(weight, 1)).item()
                    weight_of_1 = count_zero / (count_zero + count_one)
                    weight_of_0 = count_one / (count_zero + count_one)
                    weight = torch.where(weight == 1, weight_of_1, weight_of_0) #对weight矩阵赋权重
                    error_broadcast = nn.BCELoss(weight=weight, reduction='mean')  # 二元交叉损失熵

                    # 对非广播轮次的内容（全0张量）进行掩码覆盖
                    mask_broadcast = torch.all(one_batch_broadcast_label_data == 0, dim=-1)  # 判断最后一维是否全为0，生成掩码
                    broadcast_decision = torch.where(mask_broadcast.unsqueeze(-1), torch.zeros_like(broadcast_decision),
                                                     broadcast_decision)  # 将对应位置全置为0

                    loss_broadcast = error_broadcast(broadcast_decision, one_batch_broadcast_label_data) #对每一位求二分类交叉损失熵

                    # 计算mine的损失权重（使用Weighted Cross Entropy Loss）：
                    weight_2 = one_batch_mine_label_data
                    # 统计one_batch_broadcast_label_data中0的个数
                    count_zero = torch.sum(torch.eq(weight_2, 0)).item()
                    # 统计one_batch_broadcast_label_data中1的个数
                    count_non_zero = weight_2.numel() - count_zero
                    weight_of_non_0 = count_zero / (count_zero + count_non_zero)
                    weight_of_0 = count_non_zero / (count_zero + count_non_zero)
                    weight_2 = torch.where(weight_2 == 0, weight_of_0, weight_of_non_0)  # 对weight矩阵赋权重
                    error_mine = nn.BCELoss(weight=weight_2, reduction='mean')  # 二元交叉损失熵

                    # 对非挖矿轮次的内容（全0张量）进行掩码覆盖
                    mask_mine = torch.all(one_batch_mine_label_data == 0, dim=-1)  # 判断最后一维是否全为0，生成掩码
                    mine_decision = torch.where(mask_mine.unsqueeze(-1), torch.zeros_like(mine_decision),
                                                mine_decision)  # 将对应位置全置为0

                    loss_mine = error_mine(mine_decision, one_batch_mine_label_data)

                    loss = loss_broadcast + loss_mine
                    loss.backward() # average value loss
                    optimizer.step()

                    if loader == num_loader - 1:
                        print("epoch: {} ...".format(epoch))
                        print("     loss: {}".format(loss.data.item()))
                        print("     broadcast loss: {}".format(loss_broadcast.data.item()))
                        print("     mine loss: {}".format(loss_mine.data.item()))
    else:
        learning_rate = 0.0005
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        folder_path = TRAIN_DATA_PATH + '4_more_value'
        file_count = count_files_in_folder(folder_path)
        num_batches = int(file_count / BATCH_SIZE) #向下取整
        for epoch in range(int(num_epochs)):
            #23-9-11新增！！！！！！！！！！！！！！随机化训练数据读取
            epoch_file_list = os.listdir(folder_path)  # 获取文件列表
            random.shuffle(epoch_file_list)  # 原地洗牌操作

            for loader in range(num_batches):
                one_batch_train_data, one_batch_broadcast_label_data, one_batch_mine_label_data = data_load.load_one_batch_data(
                        folder_path, epoch_file_list, batch_round=loader)
                optimizer.zero_grad()

                # 对input进行0值的添加处理
                value_to_add = one_batch_train_data.mean() #求得非0数值的平均值
                '''
                
                '''
                # 对input中的1的左右的0值进行微增处理
                increment = 0.1 # 定义微小增量的值
                indices = one_batch_train_data.nonzero(as_tuple=False) # 获取稀疏张量中值为1的元素的索引
                for idx in indices:
                    plane, row, col = idx.tolist()
                    # 对左侧进行微小增量操作
                    if col > 0:
                        if one_batch_train_data[plane][row][col - 1] == 1:
                            one_batch_train_data[plane][row][col - 1] += 5*increment
                        else:
                            one_batch_train_data[plane][row][col - 1] += increment
                    # 对右侧进行微小增量操作
                    if col < one_batch_train_data.size(2) - 1:
                        if one_batch_train_data[plane][row][col + 1] == 1:
                            one_batch_train_data[plane][row][col + 1] += 5*increment
                        else:
                            one_batch_train_data[plane][row][col + 1] += increment

                one_batch_train_data = torch.add(one_batch_train_data, value_to_add)

                broadcast_decision, mine_decision = model(one_batch_train_data)

                # 计算广播的损失权重（使用Weighted Cross Entropy Loss）：
                weight = one_batch_broadcast_label_data
                # 统计one_batch_broadcast_label_data中0的个数
                count_zero = torch.sum(torch.eq(weight, 0)).item()
                # 统计one_batch_broadcast_label_data中1的个数
                count_one = torch.sum(torch.eq(weight, 1)).item()
                weight_of_1 = count_zero / (count_zero + count_one)
                weight_of_0 = count_one / (count_zero + count_one)
                weight = torch.where(weight == 1, weight_of_1, weight_of_0)  # 对weight矩阵赋权重
                error_broadcast = nn.BCELoss(weight=weight, reduction='mean')  # 二元交叉损失熵

                #对非广播轮次的内容（全0张量）进行掩码覆盖
                mask_broadcast = torch.all(one_batch_broadcast_label_data == 0, dim=-1)  # 判断最后一维是否全为0，生成掩码
                broadcast_decision = torch.where(mask_broadcast.unsqueeze(-1), torch.zeros_like(broadcast_decision),
                                            broadcast_decision)  # 将对应位置全置为0

                loss_broadcast = error_broadcast(broadcast_decision, one_batch_broadcast_label_data)  # 对每一位求二分类交叉损失熵

                # 计算mine的损失权重（使用Weighted Cross Entropy Loss）：
                weight_2 = one_batch_mine_label_data
                # 统计one_batch_broadcast_label_data中0的个数
                count_zero = torch.sum(torch.eq(weight_2, 0)).item()
                # 统计one_batch_broadcast_label_data中1的个数
                count_non_zero = weight_2.numel() - count_zero
                weight_of_non_0 = count_zero / (count_zero + count_non_zero)
                weight_of_0 = count_non_zero / (count_zero + count_non_zero)
                weight_2 = torch.where(weight_2 == 0, weight_of_0, weight_of_non_0)  # 对weight矩阵赋权重
                error_mine = nn.BCELoss(weight=weight_2, reduction='mean')  # 二元交叉损失熵

                #对非挖矿轮次的内容（全0张量）进行掩码覆盖
                mask_mine = torch.all(one_batch_mine_label_data == 0, dim=-1)  # 判断最后一维是否全为0，生成掩码
                mine_decision = torch.where(mask_mine.unsqueeze(-1), torch.zeros_like(mine_decision),
                                            mine_decision)  # 将对应位置全置为0

                loss_mine = error_mine(mine_decision, one_batch_mine_label_data)

                loss = loss_broadcast + loss_mine
                loss.backward()  # average value loss
                optimizer.step()

                print("epoch: {} - batch: {} ...".format(epoch, loader))
                print("     loss: {}".format(loss.data.item()))
                print("     broadcast loss: {}".format(loss_broadcast.data.item()))
                print("     mine loss: {}".format(loss_mine.data.item()))

        # print accuracy...

def main(train_new_model, diff_learn_rate):

    if train_new_model:
        alpha_miner = AlphaMinerModel()
    else:
        # 加载模型
        load_model_path = 'alpha_miner_12_128_20231026182749.pth'
        alpha_miner = AlphaMinerModel.load_model(TRAIN_MODEL_PATH + load_model_path)

    #输出模型参数数量
    print('Num of params = ' +str(alpha_miner.get_param_num()))
    #训练模型
    train_model(alpha_miner, diff_learn_rate)
    #存储训练好的模型
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    if 'deep' in TRAIN_DATA_PATH:
        torch.save(alpha_miner.state_dict(), TRAIN_MODEL_PATH + 'alpha_miner_deep_'+str(NUM_EPOCHS)+'_'+str(BATCH_SIZE)+'_{}.pth'.format(current_time))
    else:
        torch.save(alpha_miner.state_dict(), TRAIN_MODEL_PATH + 'alpha_miner_'+str(NUM_EPOCHS)+'_'+str(BATCH_SIZE)+'_{}.pth'.format(current_time))
    #测试模型性能

if __name__ == "__main__":
    main(train_new_model = False, diff_learn_rate = False)

