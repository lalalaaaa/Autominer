import torch
import multiprocessing

## CUDA variable from Torch
CUDA = torch.cuda.is_available()
## Dtype of the tensors depending on CUDA
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")


################################MCTS相关参数################################
SIMULATION_ROUND = 99999
#MCTS单次game最大的随机树探索“深度”模拟
MAX_ROUND_NUMBER = 15
#MCTS树横向的随机扩展“广度”模拟
COMPUTATION_BUDGET = 30
#MCTS树纵向的扩展轮次，每次都选出一个best child进入下一轮次（此值也确定了时间步的大概长度）
GAME_MAX_ROUND = 5

################################神经网络相关参数################################
#模拟次数，取平均值
TEST_ROUND = 1000
#测试LSTM网络轮数(即每轮GAME中挖矿的链的深度)：
TEST_TREE_ROUND = 3

#训练集本地地址
TRAIN_DATA_PATH = "./train_data/x-1/MCTS_output_new/"

#model保存路径
TRAIN_MODEL_PATH = "save_model/"

#epoch数量
NUM_EPOCHS = 8
#batch size
BATCH_SIZE = 128
#每个epoch训练的batch数
NUM_LOADER = 8

START_BROADCAST_RATE = 0.5

#block的最大index
MAX_BLOCK = 32
#隐藏层数量
NUM_HIDDEN = 1
#神经网络输入的window size，即，miner能看到的local_chain和global_chain的滑动窗口的长度。
INPUT_WINDOW_SIZE = 30
#LSTM神经网络的时间步数
LSTM_TIME_STEP = 32
#embed预留的输入长度
EMBED_RESERVED_LEN = 10
#embed的嵌入维度（即，nn.Embedding()的第二个参数）
EMBED_SIZE = 5
#整个网络的input dim
INPUT_FEATURE_DIM = 2*MAX_BLOCK*MAX_BLOCK
#EMBED dim
EMBED_DIM = 4
#LSTM网络的输入输出的长度（一个时间步）
LSTM_INPUT_FEATURE_DIM = INPUT_FEATURE_DIM # *EMBED_DIM if embed input
LSTM_OUTPUT_FEATURE_DIM = MAX_BLOCK + MAX_BLOCK

################################miner相关参数################################
## output数据action的格式、长度等
OUTPUT_BROADCAST_LEN = 10
OUTPUT_MINE_BLCOK_LEN = 10
OUTPUT_HP_RATE_LEN = 10
#模拟的挖矿轮次
MINE_ROUND = 1
#挖矿难度，即，一次hash运算挖矿成功的概率
HASH_DIFFICULTY = 0.00001
#参与挖矿的非本地（其他）节点的数量
NODE_NUMBER = 100
#本地挖矿算力
LOCAL_HASH_POWER = 35
#非本地挖矿算力
GLOBAL_HASH_POWER = 100 - LOCAL_HASH_POWER
#hash计算成本，即，进行一次hash运算所需的消耗
PER_HASH_COST = 0.00001
#块奖励
BLOCK_REWARD = 1
#随机挖矿父区块的随机选择窗口大小
RANDOM_FATHER_WINDOW_SIZE = 5
#最小广播数量（保证至少广播这么多区块，除非未广播区块数量不超过此数）
MIN_BROADCAST_NUM = 5
#在原有基础上(随机从[0，1，2，...,len(未广播序列)]中挑选一个数字)，增加NOT_BROADCAST_RATE%的概率不广播
NOT_BROADCAST_RATE = 30