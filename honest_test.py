import random
from const import *
import numpy as np

honest_run_times = 10000 #1w次采样
base_line = LOCAL_HASH_POWER * GAME_MAX_ROUND * BLOCK_REWARD / (LOCAL_HASH_POWER + GLOBAL_HASH_POWER) + 1

honest_value_list = [0]*16 #记录各种（+1~+16）的奖励值的次数

for i in range(honest_run_times):
    print('Round ' + str(i) + ' test ...')
    wallet_value_tmp = 0
    for _ in range(GAME_MAX_ROUND):
        # 计算local为honest时在模拟情况下的收益
        sample_local = np.random.geometric(HASH_DIFFICULTY * (LOCAL_HASH_POWER), size=1)  # 表示本地节点的挖矿时长
        samples_other = np.random.geometric(HASH_DIFFICULTY * (GLOBAL_HASH_POWER /NODE_NUMBER), size=NODE_NUMBER)  # 表示其他节点的挖矿时长列表
        if sample_local[0] < min(samples_other):
            wallet_value_tmp += 1
    if wallet_value_tmp >= base_line and wallet_value_tmp < base_line + 1:
        honest_value_list[0] += 1
    elif wallet_value_tmp >= base_line + 1 and wallet_value_tmp < base_line + 2:
        honest_value_list[1] += 1
    elif wallet_value_tmp >= base_line + 2 and wallet_value_tmp < base_line + 3:
        honest_value_list[2] += 1
    elif wallet_value_tmp >= base_line + 3 and wallet_value_tmp < base_line + 4:
        honest_value_list[3] += 1
    elif wallet_value_tmp >= base_line + 4 and wallet_value_tmp < base_line + 5:
        honest_value_list[4] += 1
    elif wallet_value_tmp >= base_line + 5 and wallet_value_tmp < base_line + 6:
        honest_value_list[5] += 1
    elif wallet_value_tmp >= base_line + 6 and wallet_value_tmp < base_line + 7:
        honest_value_list[6] += 1
    elif wallet_value_tmp >= base_line + 7 and wallet_value_tmp < base_line + 8:
        honest_value_list[7] += 1
    elif wallet_value_tmp >= base_line + 8 and wallet_value_tmp < base_line + 7:
        honest_value_list[8] += 1

print(honest_value_list) # 输出honest_value_list