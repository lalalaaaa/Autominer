import pickle
import miner
#import pprint
import torch
import torch.nn as nn
import numpy as np
from const import *
#读取数据相关
import os
import pickle
import random

############################################
############################################
###############DATA PROCESS FUNCTION########
############################################
############################################

# process broadcast action list
def broadcast_to_multi_hot(index_array, num_classes=MAX_BLOCK):
    # 将index数组变换为multi-hot数组
    multi_hot_vector = torch.zeros(num_classes)
    filtered_array = [x for x in index_array if x < num_classes] #将原始数组中超过num_classes的数字（不合法）删掉
    if sum(filtered_array) > 0: #判断index_array是否为全0向量
        multi_hot_vector[filtered_array] = 1
    return torch.tensor(multi_hot_vector).flatten()

# process father&rate action list
def father_and_rate_to_multi_rate(rate_array, father_array, num_classes=MAX_BLOCK):
    # 将rate数组变换为multi-rate数组
    multi_rate_vector = torch.zeros(num_classes)
    unique_father_array = list(set(father_array))  # 去除重复的father_array元素
    for father in unique_father_array:
        indices = [i for i, f in enumerate(father_array) if f == father]  # 找到father_array中等于当前father的索引
        sum_value = sum([rate_array[i] for i in indices])  # 对相应索引处的rate_array进行求和
        if father < num_classes: #防止out of limit
            multi_rate_vector[father] = sum_value
    multi_rate_vector = multi_rate_vector[:num_classes] #固定其输出为num_classes长度
    return torch.tensor(multi_rate_vector).flatten()

# process 4 chains-1
def chain_to_adj_matrix(index_chain, index_father, num_classes=MAX_BLOCK): # 将DAG-chain转换为邻接矩阵的形式
    # 创建邻接矩阵
    adj_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(index_chain)):
        if index_chain[i] < num_classes and index_father[i] < num_classes:
            # 在邻接矩阵中标记边的关系
            adj_matrix[index_chain[i], index_father[i]] = 1
    return adj_matrix
# process 4 chains-2
def adj_matrix_to_1nn(adj_matrix): # 将n*n的邻接矩阵变换为1*(n*n)的一维矩阵
    return torch.tensor(adj_matrix.reshape(1, -1)).flatten() # 1 表示结果矩阵的行数，而 -1 表示自动计算列数以匹配原始矩阵中的总元素个数。

def split_tensor(input_tensor): #因为计算loss时是将broadcast和mine分开计算的，因此需要对label张量进行切片处理
    # 切片操作，取前64个通道和后64个通道
    tensor1 = input_tensor[:, :, :MAX_BLOCK]
    tensor2 = input_tensor[:, :, MAX_BLOCK:2*MAX_BLOCK]
    return tensor1, tensor2


#################################################################

# op-process multi_hot to broadcast action list
def multi_hot_to_broadcast(multi_hot_vector, output_len=OUTPUT_BROADCAST_LEN): # output_len = OUTPUT_BROADCAST_LEN
    index_array = torch.nonzero(multi_hot_vector).squeeze()
    # 根据输出数组的长度进行截断或填充
    if len(index_array) < output_len:
        index_array = torch.cat([index_array, torch.zeros(output_len - len(index_array), dtype=torch.long)])
    elif len(index_array) > output_len:
        index_array = index_array[:output_len]
    return index_array

# op-process multi_rate to father&rate action list
def multi_rate_to_father_and_rate(multi_rate_vector, output_len=OUTPUT_MINE_BLCOK_LEN): # output_len = OUTPUT_MINE_BLCOK_LEN
    nonzero_indices = torch.nonzero(multi_rate_vector).squeeze()
    num_nonzero = len(nonzero_indices)

    # 根据输出数组的长度进行截断或填充
    if num_nonzero < output_len:
        rate_array = torch.cat([multi_rate_vector[nonzero_indices], torch.zeros(output_len - num_nonzero)])
        father_array = torch.cat([nonzero_indices, torch.zeros(output_len - num_nonzero, dtype=torch.long)])
    elif num_nonzero > output_len:
        rate_array = multi_rate_vector[nonzero_indices[:output_len]]
        father_array = nonzero_indices[:output_len]
    else:
        rate_array = multi_rate_vector[nonzero_indices]
        father_array = nonzero_indices

    return rate_array, father_array

########################################################
########################################################

def process_one_time_step_data(org_one_time_step_data):
    one_time_step_train_data = []
    one_time_step_label_data = []

    # 初始化one_time_step_train_data
    init_local_DAGchain = [0] * INPUT_WINDOW_SIZE
    init_local_DAGchain_father = [0] * INPUT_WINDOW_SIZE
    init_global_DAGchain = [0] * INPUT_WINDOW_SIZE
    init_global_DAGchain_father = [0] * INPUT_WINDOW_SIZE
    init_local_adj_matrix = chain_to_adj_matrix(init_local_DAGchain, init_local_DAGchain_father)
    init_local_adj_1nn = adj_matrix_to_1nn(init_local_adj_matrix)  # final input type
    init_global_adj_matrix = chain_to_adj_matrix(init_global_DAGchain, init_global_DAGchain_father)
    init_global_adj_1nn = adj_matrix_to_1nn(init_global_adj_matrix)  # final input type
    one_time_step_train_data.append(torch.cat((init_local_adj_1nn, init_global_adj_1nn), dim=0))

    for i,(org_data_1, org_data_2) in enumerate(org_one_time_step_data): #org_data_1 is needembed data, org_data_2 is noneedembed data
        father_list = org_data_1[-OUTPUT_MINE_BLCOK_LEN:]
        broadcast_list = org_data_1[-(OUTPUT_MINE_BLCOK_LEN + OUTPUT_BROADCAST_LEN):-OUTPUT_MINE_BLCOK_LEN]
        rate_list = org_data_2

        broadcast_multi_hot = broadcast_to_multi_hot(broadcast_list)  # final input type
        mine_multi_rate = father_and_rate_to_multi_rate(rate_list, father_list)  # final input type


        local_DAGchain = org_data_1[:INPUT_WINDOW_SIZE]
        local_DAGchain_father = org_data_1[INPUT_WINDOW_SIZE:2*INPUT_WINDOW_SIZE]
        global_DAGchain = org_data_1[2*INPUT_WINDOW_SIZE:3*INPUT_WINDOW_SIZE]
        global_DAGchain_father = org_data_1[3*INPUT_WINDOW_SIZE:4*INPUT_WINDOW_SIZE]

        local_adj_matrix = chain_to_adj_matrix(local_DAGchain,local_DAGchain_father)
        local_adj_1nn = adj_matrix_to_1nn(local_adj_matrix) #final input type
        global_adj_matrix = chain_to_adj_matrix(global_DAGchain, global_DAGchain_father)
        global_adj_1nn = adj_matrix_to_1nn(global_adj_matrix) #final input type

        one_time_step_train_data.append(torch.cat((local_adj_1nn, global_adj_1nn), dim=0))
        #if i != 0: #label不需要第一组原始action数据
        #不用去掉第一组数据，因为现在的input已经只有chain的state了，没有action-23.7.28
        one_time_step_label_data.append(torch.cat((broadcast_multi_hot, mine_multi_rate), dim=0))

    one_time_step_train_data = one_time_step_train_data[:-1] #删除train data中的最后一个元素，因为不需要，最开始初始化补0了

    # 补齐LSTM TIME STEP长度
    if len(one_time_step_train_data) < LSTM_TIME_STEP:
        while len(one_time_step_train_data) < LSTM_TIME_STEP:
            one_time_step_train_data.append(torch.zeros(INPUT_FEATURE_DIM))
    elif len(one_time_step_train_data) > LSTM_TIME_STEP:
        one_time_step_train_data = one_time_step_train_data[:LSTM_TIME_STEP]

    if len(one_time_step_label_data) < LSTM_TIME_STEP:
        while len(one_time_step_label_data) < LSTM_TIME_STEP:
            one_time_step_label_data.append(torch.zeros(LSTM_OUTPUT_FEATURE_DIM))
    elif len(one_time_step_label_data) > LSTM_TIME_STEP:
        one_time_step_label_data = one_time_step_label_data[:LSTM_TIME_STEP]

    return torch.stack(one_time_step_train_data), torch.stack(one_time_step_label_data)

def load_one_batch_data(_folder_path, epoch_file_list, batch_round, batch_size=BATCH_SIZE):
    # 定义存储训练数据的数组
    one_batch_train_data = []
    one_batch_label_data = []
    folder_path = _folder_path
    file_list = epoch_file_list

    start_index = batch_round * batch_size
    end_index = (batch_round+1) * batch_size
    selected_files = file_list[start_index:end_index] #按顺序选择文件

    #selected_files = random.sample(file_list, batch_size) #随机选择文件

    count = 0
    # 遍历选中的文件
    for file_name in selected_files:
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        # 使用pickle模块读取文件中的数据对象
        with open(file_path, 'rb') as file:
            one_time_step_train_data = pickle.load(file)
        processed_one_time_step_train_data, processed_one_time_step_label_data = process_one_time_step_data(one_time_step_train_data)
        one_batch_train_data.append(processed_one_time_step_train_data)
        one_batch_label_data.append(processed_one_time_step_label_data)
        count += 1
        if count >= batch_size:
            break
    one_batch_label_data = torch.stack(one_batch_label_data)
    one_batch_train_data = torch.stack(one_batch_train_data)
    one_batch_broadcast_label_data, one_batch_mine_label_data = split_tensor(one_batch_label_data)
    return one_batch_train_data.float(), one_batch_broadcast_label_data.float(), one_batch_mine_label_data.float()

def main():
    one_batch_train_data, one_batch_broadcast_label_data, one_batch_mine_label_data = load_one_batch_data(TRAIN_DATA_PATH)
    print(one_batch_train_data)
    print(one_batch_train_data.shape)  # shape: (batch_size=32, time step=32, 4096*2+64*2=8320)
    print(one_batch_broadcast_label_data)
    print(one_batch_broadcast_label_data.shape)  # shape: (batch_size=32, time step=32, 64)
    print(one_batch_mine_label_data)
    print(one_batch_mine_label_data.shape)  # shape: (batch_size=32, time step=32, 64)

if __name__ == "__main__":
    main()




'''
############################################
############################################
#################处理father和rate############
############################################
############################################

def process_father_and_rate(father, rate):
    # Create a dictionary to store the sum of rates for each unique father element
    father_dict = {}

    # Iterate through father and rate arrays
    for i in range(len(father)):
        val = father[i]
        if val not in father_dict:
            father_dict[val] = rate[i]
        else:
            father_dict[val] += rate[i]

    # Sort the dictionary by keys (father elements) in descending order
    sorted_dict = dict(sorted(father_dict.items(), key=lambda x: x[0], reverse=True))

    # Extract sorted father elements and their corresponding summed rates
    sorted_father = list(sorted_dict.keys())
    sorted_rate = list(sorted_dict.values())

    # Append zeros to make the length of father and rate arrays equal to action size
    sorted_father.extend([0] * (len(rate) - len(sorted_father)))
    sorted_rate.extend([0] * (len(rate) - len(sorted_rate)))

    return sorted_father, sorted_rate

############################################
############################################
#################读取数据####################
############################################
############################################

#max_length相当于是LSTM中的time step
max_length = GAME_MAX_ROUND * 2 + 10 # 由于每轮GAME节点产生的行为数量不固定，因为定义一个行为次数的上限，保证每个训练数据和label数据的形状都是相同的。

# 定义存储训练数据的数组
train_data_array = []
label_data_tensor = [] #模型的label数据array
# 指定训练数据文件夹路径
folder_path = "MCTS_output_test/"
# 遍历文件夹中的文件
for file_name in os.listdir(folder_path):
    # 检查文件名是否以"output_LSTM_data_high_value_"为前缀
    if file_name.startswith("output_LSTM_data_high_value_") or file_name.startswith("output_LSTM_data_2_value_1_round_"):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        # 使用pickle模块读取文件中的数据对象
        with open(file_path, 'rb') as file:
            train_data = pickle.load(file)

        tmp_labels = []
        for i,(need_embed_data, noneed_embeded_data) in enumerate(train_data):
            father_list = need_embed_data[-OUTPUT_MINE_BLCOK_LEN:]
            new_father, new_rate = process_father_and_rate(father_list, noneed_embeded_data)
            new_need_embed_data = need_embed_data[:-OUTPUT_MINE_BLCOK_LEN] + new_father
            new_noneed_embeded_data = new_rate
            if i != 0 and i < max_length: # 注意，不需要第一组行为数据
                broadcast_list = need_embed_data[-(OUTPUT_MINE_BLCOK_LEN + OUTPUT_BROADCAST_LEN):-OUTPUT_MINE_BLCOK_LEN]
                tmp_labels.append(torch.tensor(broadcast_list + new_father + new_rate))
            train_data[i] = (new_need_embed_data, new_noneed_embeded_data)

        # 将读取的数据对象添加到数组中
        train_data_array.append(train_data)
        # 补0以保证每轮的label_data的shape一致
        if len(tmp_labels) < max_length:
            # 计算需要插入的0的行数
            num_zeros = max_length - len(tmp_labels)
            # 创建一个形状为(num_zeros, feature_dim)的全零数组
            zeros = torch.zeros((num_zeros, (OUTPUT_MINE_BLCOK_LEN + OUTPUT_BROADCAST_LEN + OUTPUT_HP_RATE_LEN)))
            # 将全零数组添加到one_batch_input_data中
            label_data_tensor.append(torch.cat((torch.stack(tmp_labels), zeros), dim=0))

# 堆叠张量可将list=[tensor1（维度为n）, tensor2, tensor3, ..., ]变成一个大的tensor（维度为n+1）
label_data_tensor = torch.stack(label_data_tensor)

############################################
############################################
#################处理数据####################
############################################
############################################

embedding = nn.Embedding(INPUT_WINDOW_SIZE + 10, EMBED_SIZE)

#((4 * INPUT_WINDOW_SIZE + OUTPUT_BROADCAST_LEN + OUTPUT_MINE_BLCOK_LEN) * EMBEDED_SIZE + OUTPUT_HP_RATE_LEN)
input_data = [] #shape: (batch_dim, seq_dim, feature_dim)
for i, train_data in enumerate(train_data_array):
    #每轮循环都是一个完整时间序列
    one_batch_input_data = []

    for j,(need_embedded_input, noneed_embedded_input) in enumerate(train_data):
        if j < max_length: #超过最大上限的则不计入
            #每个循环都是单个数据
            embedded_input = embedding(torch.tensor(need_embedded_input)) #需要embed的数据
            # 使用flatten()函数将二维数组展平为一维数组
            embedded_input_flat = embedded_input.flatten()
            # 拼接两个张量
            one_batch_input_data = torch.cat((embedded_input_flat.unsqueeze(0), torch.tensor(noneed_embedded_input).unsqueeze(0)), dim=1)

    # 补0以保证每轮的one_batch_input_data的shape一致
    if len(one_batch_input_data) < max_length:
        # 计算需要插入的0的行数
        num_zeros = max_length - len(one_batch_input_data)
        # 创建一个形状为(num_zeros, feature_dim)的全零数组
        zeros = torch.zeros((num_zeros, ((4 * INPUT_WINDOW_SIZE + OUTPUT_BROADCAST_LEN + OUTPUT_MINE_BLCOK_LEN) * EMBED_SIZE + OUTPUT_HP_RATE_LEN)))
        # 将全零数组添加到one_batch_input_data中
        one_batch_input_data = torch.cat((one_batch_input_data, zeros), dim=0)

    input_data.append(one_batch_input_data)

#for i,label_data in enumerate(label_data_array):
    # 每轮循环都是一个完整时间序列
    #one_batch_label_data = []
    #for j,(broadcast, father, rate) in enumerate(label_data):

# 将input_data中的张量进行堆叠
input_data_tensor = torch.stack(input_data, dim=0)

print(input_data_tensor)
print(input_data_tensor.shape) # shape: (batch_size, max_length(time step), 140*5+10=710)
print(label_data_tensor)
print(label_data_tensor.shape) # shape: (batch_size, max_length(time step), 30)

#################不省略地打印张量###############
# 将张量转换为NumPy数组
#array = label_data_tensor.numpy()
# 设置打印选项
#np.set_printoptions(threshold=np.inf)  # 或者使用 sys.maxsize 代替 np.inf
# 打印数组
#print(array)
'''