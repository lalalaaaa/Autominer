import copy
import random
from const import *
import numpy as np
from collections import defaultdict

class Block:
    def __init__(self, bit = False, broadcasted = True, index = 0, hash_time = 0, father_index = 0):
        if not isinstance(bit, bool):
            raise ValueError('bit must be a boolean value')
        if not isinstance(broadcasted, bool):
            raise ValueError('broadcasted must be a boolean value')
        self.bit = bit
        self.broadcasted = broadcasted
        self.index = index #此index是在本地节点视角中的区块编号, 此参数只对本地节点有效
        self.next = []
        self.hash_time = hash_time #表示本块被挖掘出来所花费的时间，创世块此值为0
        self.father_index = father_index #表示此区块的父区块，创世块此值为0
    
    def get_bit(self):
        return self.bit
    def get_broadcasted(self):
        return self.broadcasted
    def get_index(self):
        return self.index
    def get_next(self):
        return self.next
    def get_hash_time(self):
        return self.hash_time
    def get_father_index(self):
        return self.father_index
    def set_father_index(self, parm):
        self.father_index = parm

class Chain:
    def __init__(self):
        self.head = None
        self.blocks_by_index = {}  # 块的index字典
        self.chain_length = 0

    def add_block(self, block, father_index = -1): #father_index为添加区块的父区块索引
        if not isinstance(block, Block):
            raise ValueError('block must be an instance of Block')
        if not self.head:
            block.set_father_index(father_index)
            self.head = block
        elif father_index >= 0:
            block.set_father_index(father_index)
            self.blocks_by_index.get(father_index).next.append(block) #根据父区块索引将此区块放入其子区块集合（next）中
        else:
            raise ValueError('must input valid father_index')
        self.blocks_by_index[block.index] = block  # 将新区块添加到字典中
        self.chain_length += 1 # 将链的长度+1

    def get_block_by_index(self, index):  # 根据index快速找到索引的块
        return self.blocks_by_index.get(index)

    def get_chain_index(self): #当要添加块时，获取添加的此块的index值
        return self.chain_length

    def print_chain(self):
        def _print(block, indent=0):
            print('  ' * indent + '├──' + f' Block Index: {block.index}, Bit: {block.bit}, Broadcasted: {block.broadcasted}')
            for child in block.next:
                _print(child, indent + 1)
        print('Root')
        _print(self.head)
        print('chain length is ' + str(self.chain_length))

    def get_lastest_block(self): #此方法仅在Chain为无分支链时调用
        if not self.head:
            return None
        current = self.head
        while current.next:
            current = current.next[0]  # 假设每个区块至少有一个子节点
        return current

    def get_chain_len(self): #获取此chain（DAG结构）的所有节点的数量
        def _count_nodes(block):
            return 1 + sum(_count_nodes(child) for child in block.next)
        return _count_nodes(self.head) if self.head else 0

    # 获取未广播的块的索引index list
    def get_unbroadcasted_block_list(self):
        tmp_list = []
        tmp_head = self.head

        def check_block(_block):
            if _block.next:
                if not _block.broadcasted:
                    tmp_list.append(_block.index)
                for child in _block.next:
                    check_block(child)
            else:
                if not _block.broadcasted:
                    tmp_list.append(_block.index)

        check_block(tmp_head)
        return tmp_list

    #获取符合LSTM的链状态的输出（2*window_size数组）
    def chain_4_LSTM_input(self, window_size = INPUT_WINDOW_SIZE):
        target_indices = [0] * window_size
        target_fathers = [0] * window_size

        def traverse_chain(block):
            nonlocal target_indices, target_fathers

            index = block.get_index()
            father_index = 0 if index == 0 else block.get_father_index()

            target_indices.append(index)
            target_fathers.append(father_index)

            if len(target_indices) > window_size:
                min_index = min(target_indices)
                min_idx = target_indices.index(min_index)

                target_indices.pop(min_idx)
                target_fathers.pop(min_idx)

            for child in block.get_next():
                traverse_chain(child)

        if self.head is not None:
            traverse_chain(self.head)

        return target_indices, target_fathers

class NodeAction:
    #目前设定的action list的格式是在每轮（块与块之间）
    def __init__(self, _broadcast_blocks_list = [0]*OUTPUT_BROADCAST_LEN, _hp_rate_list = [0]*OUTPUT_HP_RATE_LEN, _mine_father_blocks_list = [0]*OUTPUT_MINE_BLCOK_LEN):
        self.hp_rate_list = _hp_rate_list
        self.mine_father_blocks_list = _mine_father_blocks_list
        self.broadcast_blocks_list = _broadcast_blocks_list
        #self.broadcast_blocks_list_after = _broadcast_blocks_list_after

    def get_hp_rate_list(self):
        return self.hp_rate_list
    
    def set_hp_rate_list(self, parm):
        if len(parm) == OUTPUT_HP_RATE_LEN:
            self.hp_rate_list = parm
            #print('Local Node HP rate = ' + str(self.get_hp_rate_list()))
        else:
            raise ValueError("Invalid input length. Expected a list of length 10.")
    
    def get_mine_father_blocks_list(self):
        return self.mine_father_blocks_list
    
    def set_mine_father_blocks_list(self, parm):
        if len(parm) == OUTPUT_MINE_BLCOK_LEN:
            self.mine_father_blocks_list = parm
            #print('Local Node mine father blocks = ' + str(self.get_mine_father_blocks_list()))
        else:
            raise ValueError("Invalid input length. Expected a list of length 10.")
    
    def get_broadcast_blocks_list(self):
        return self.broadcast_blocks_list
    
    def set_broadcast_blocks_list(self, parm):
        if len(parm) == OUTPUT_BROADCAST_LEN:
            self.broadcast_blocks_list = parm
            #print('Local Node broadcast blocks list = ' + str(self.get_broadcast_blocks_list()))
        else:
            raise ValueError("Invalid input length. Expected a list of length 20.")

    #检测此action list在当前状态下（local_chain, consensus_chain）是否合法,
    #并返回错误行为的位置索引的列表
    def ver_broadcast_blocks_list_before(self, local_chain):
        unbroadcasted_block_list = local_chain.get_unbroadcasted_block_list() #返回未广播block的index list
        err_action_list = [] #用于存储broadcast_blocks_list_before中错误input的位置
        for i, index in enumerate(self.broadcast_blocks_list_before):
            if index not in unbroadcasted_block_list:
                err_action_list.append(i)
        return err_action_list

    def ver_hp_rate_list_and_mine_father_blocks_list(self, local_chain):
        #检测hp_err_action_list，并将不合法的位置挑出来。
        hp_err_action_list = []
        for i, rate in enumerate(self.hp_rate_list):
            if rate < 0:
                hp_err_action_list.append(i)
        if sum(self.hp_rate_list) != 1:
            hp_err_action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #检测mine_err_action_list，并将不合法的位置挑出来。
        mine_err_action_list = []
        for i, index in enumerate(self.mine_father_blocks_list):
            if index not in range(local_chain.get_chain_length()):
                mine_err_action_list.append(i)
        return hp_err_action_list, mine_err_action_list
    
    def generate_random_broadcast_action(self, world_state):
        #随机生成合法的广播行为的output list（before）。
        unbroadcasted_block_list = world_state.Local_chain.get_unbroadcasted_block_list() #返回未广播block的index list
        if random.random() > NOT_BROADCAST_RATE/100:
            tmp_broadcast_block_num = random.randint(0, len(unbroadcasted_block_list)) # 随机地确定本轮要广播多少块（有可能超过OUTPUT_BROADCAST_LEN上界的合法值）
        else:
            tmp_broadcast_block_num = 0 #在原有基础上，增加NOT_BROADCAST_RATE%的概率不广播
        #tmp_broadcast_block_num = min(len(unbroadcasted_block_list), max(MIN_BROADCAST_NUM, tmp_broadcast_block_num)) #保证至少广播MIN_BROADCAST_NUM个区块
        broadcast_number = min(tmp_broadcast_block_num, OUTPUT_BROADCAST_LEN) # 随机地确定本轮要广播多少块（不超过OUTPUT_BROADCAST_LEN上界的合法值）
        output_broadcast = [0] * (OUTPUT_BROADCAST_LEN - broadcast_number) # 确定不广播的数量，0表示不广播
        broadcast_elements = random.sample(unbroadcasted_block_list, broadcast_number)
        output_broadcast.extend(broadcast_elements)
        random.shuffle(output_broadcast) #将列表output_broadcast进行随机打乱顺序。
        self.set_broadcast_blocks_list(output_broadcast) #对self的broadcast_blocks_list进行赋值。
        return output_broadcast

    def generate_random_mine_rate_action(self):
        #随机生成合法的挖矿算力分配output list。
        output_rate = [random.uniform(0, 1) for _ in range(OUTPUT_HP_RATE_LEN)]
        output_rate_sum = sum(output_rate)
        output_rate = [rate/output_rate_sum for rate in output_rate]
        self.set_hp_rate_list(output_rate)
        return output_rate
    
    def generate_random_father_block_action(self, world_state):
        #随机生成合法的挖矿父区块编号output list。
        # Create an empty list to store the output_father_block_list
        output_father_block_list = []
        tmp_parm = world_state.Local_chain.get_chain_len()
        local_chain_index = list(range(max(0, tmp_parm - RANDOM_FATHER_WINDOW_SIZE), tmp_parm))
        # Generate the output_father_block_list array
        for _ in range(OUTPUT_MINE_BLCOK_LEN):
            # Calculate the probability of selecting each element based on its index
            probability = 1 / (len(local_chain_index) - output_father_block_list.count(None))
            # Create a list of indices to choose from, with higher probabilities for later indices
            indices = list(range(len(local_chain_index)))
            indices.extend([None] * output_father_block_list.count(None))
            #选中的概率随着编号的靠后而线性增大！！！注意！random.choices()函数是有放回采样！！！。
            selected_index = random.choices(indices, weights=[probability] * len(indices))[0]
            # Add the selected element to the output_father_block_list
            output_father_block_list.append(local_chain_index[selected_index])
        self.set_mine_father_blocks_list(output_father_block_list)
        return output_father_block_list
    
    #合并挖矿算力分配和挖矿父块的“同类项”
    def combine_rate_and_father(self):
        position_dict = defaultdict(float)
        arr1 = self.get_mine_father_blocks_list()
        arr2 = self.get_hp_rate_list()
        for i, x in enumerate(arr1):
            position_dict[x] += arr2[i]
        result_dict = {x: position_dict[x] for x in position_dict}
        return result_dict

#全局变量
Genesis_block = Block(False, True, 0) #创世块
Global_time = 0 #全局时间戳
Tnx_Pool = [] #交易池
Tnx_fee = 0 #交易费

class WorldState: #世界状态，用于存储本地链和全局链
    def __init__(self):
        self.Local_chain = Chain() #本地链(未被剪枝的，本地节点视角下的)
        self.Local_chain.add_block(copy.deepcopy(Genesis_block)) #注意！这里一定要使用深拷贝，不然会所有的链都指向同一个创世块！！！
        self.Global_chain = Chain() #全局最长链（被剪枝的，全网共识的）
        self.Global_chain.add_block(copy.deepcopy(Genesis_block)) #注意！这里一定要使用深拷贝，不然会所有的链都指向同一个创世块！！！
    
    def deep_copy(self):
        # Create a new instance of the WorldState class
        copied_worldstate = WorldState()
        copied_worldstate.Local_chain = copy.deepcopy(self.Local_chain)
        copied_worldstate.Global_chain = copy.deepcopy(self.Global_chain)
        return copied_worldstate

    def get_global_broadcasted_longest_chain(self):
        #剪枝掉所有未广播的区块
        def prune_non_broadcasted_blocks(Gchain):
            def _prune(block):
                pruned_next = [child for child in block.next if child.broadcasted]
                block.next = pruned_next
                for child in pruned_next:
                    _prune(child)
            # Create a deep copy of the Global_chain
            Pruned_chain = copy.deepcopy(Gchain)
            _prune(Pruned_chain.head)
            return Pruned_chain

        #使用深度优先算法获取最长的无分支链条
        def get_longest_chain(chain):
            # 使用DFS找出从每个节点开始的最长链
            def dfs(node):
                nonlocal max_length, longest_chain
                visited.add(node)
                current_chain.append(node)
                if len(current_chain) > max_length:
                    max_length = len(current_chain)
                    longest_chain = list(current_chain)
                for child in node.next:
                    if child not in visited:
                        dfs(child)
                current_chain.pop()
                visited.remove(node)
            visited = set()
            current_chain = []
            longest_chain = []
            max_length = 0
            dfs(chain.head)
            #print('longest_chain is :')
            #print(str(longest_chain))
            # Create a new Chain object and add the nodes of the longest chain to it
            longest_chain_chain = Chain()
            for i in range(len(longest_chain)):
                if i == 0:  # The first node in the chain is the head node
                    tmp_block = Block(longest_chain[i].bit, longest_chain[i].broadcasted, longest_chain[i].index)
                    longest_chain_chain.add_block(tmp_block)
                    #print('added block: ' + str(longest_chain[i]))
                    #print('longest_chain_chain is :')
                    #longest_chain_chain.print_chain()
                else:  # For other nodes, add them to the chain with the previous node as the father node
                    tmp_block = Block(longest_chain[i].bit, longest_chain[i].broadcasted, longest_chain[i].index)
                    longest_chain_chain.add_block(tmp_block, longest_chain[i-1].index)
                    #print('added block: ' + str(longest_chain[i]))
                    #print('longest_chain_chain is :')
                    #longest_chain_chain.print_chain()
            return longest_chain_chain

        # 剪枝未广播的区块
        pruned_chain = prune_non_broadcasted_blocks(self.Local_chain)
        # 获得被广播最长链的主函数
        tmp_longest_chain = get_longest_chain(pruned_chain)
        return tmp_longest_chain

    def refresh_longest_chain(self): #返回一个布尔值，表示Global_chain是否被更新
        tmp_longest_chain = self.get_global_broadcasted_longest_chain()
        if tmp_longest_chain.get_chain_len() <= self.Global_chain.get_chain_len():
            return False
        else:
            #print('refresh Global_chain !')
            self.Global_chain = tmp_longest_chain
            #self.Global_chain.print_chain()
            return True

class OtherNode: #非本地节点
    def __init__(self, global_time, hash_power, cost, wallet, mine_father_block = 0):
        self.global_time = global_time
        self.hash_power = hash_power
        self.cost = cost
        self.wallet = wallet
        self.mine_father_block = mine_father_block #在哪个块上挖下一个块
    
    #获取节点cost值
    def get_cost(self):
        return self.cost
    #设置节点cost值
    def set_cost(self, parm):
        self.cost = parm

    #获取节点mine_father_block值
    def get_mine_father_block(self):
        return self.mine_father_block
    #设置节点mine_father_block值
    def set_mine_father_block(self, parm):
        self.mine_father_block = parm #挖掘块的index编号
    
    def refresh_action(self):
        pass

class Node: #本地节点
    def __init__(self, global_time, hash_power, cost, wallet):
        #self.index = index
        self.global_time = global_time
        self.hash_power = hash_power
        self.cost = cost
        self.wallet = wallet
        self.actions = []
        tmp_init_action = NodeAction()
        self.actions.append(tmp_init_action)
        self.lastest_action = False #本值记录最新的动作属性：False表示挖矿，True表示广播

    #获取节点cost值
    def get_cost(self):
        return self.cost
    #设置节点cost值
    def set_cost(self, parm):
        self.cost = parm
    
    #计算并返回钱包值
    def get_wallet(self, world_state):
        self.compute_wallet(world_state)
        return self.wallet
    #设置节点wallet值
    def set_wallet(self, parm):
        self.wallet = parm

    def get_hash_power(self):
        return self.hash_power

    #获取节点action列表
    def get_actions(self):
        return self.actions
    def get_broadcast_action(self):
        return self.actions[-1].get_broadcast_blocks_list()
    def get_mine_rate_action(self):
        return self.actions[-1].get_hp_rate_list()
    def get_mine_father_action(self):
        return self.actions[-1].get_mine_father_blocks_list()
    
    #设置节点action的一系列函数
    def add_one_action(self, parm):
        self.actions.append(parm)

        '''
    def set_actions_mine(self, _hp_rate_list, _mine_father_blocks_list):
        self.actions.hp_rate_list = _hp_rate_list
        self.actions.mine_father_blocks_list = _mine_father_blocks_list
    def set_actions_broadcast(self, _broadcast_blocks_list):
        self.actions.broadcast_blocks_list = _broadcast_blocks_list
        '''
    def init_mine_action(self, world_state, link_list): # 参数类型：class: WorldState ,NodeAction
        init_node_action = NodeAction(_hp_rate_list = [0.777,0.333,0,0,0,0,0,0,0,0]) #初始化动作，全部算力用于基于创世块挖掘
        self.add_one_action(init_node_action)
        link_list.append((copy.deepcopy(world_state), init_node_action))

    def random_set_action_mine(self, world_state, link_list): # 参数类型：class: WorldState ,NodeAction
        #用tmp来作为生成器生成随机action（调用generate_random_broadcast_action函数）
        tmp1 = copy.deepcopy(self.actions[-1])
        tmp2 = tmp1.generate_random_father_block_action(world_state)
        tmp3 = tmp1.generate_random_mine_rate_action()
        #记录本轮action
        new_node_action = NodeAction(_hp_rate_list=tmp3, _mine_father_blocks_list=tmp2)
        self.add_one_action(new_node_action)
        #记录本轮的action和world state的link
        link_list.append((copy.deepcopy(world_state), new_node_action))
        #self.lastest_action = 0

    def random_set_action_broadcast(self, world_state, link_list):
        #用tmp来作为生成器生成随机action（调用generate_random_broadcast_action函数）
        tmp1 = copy.deepcopy(self.actions[-1])
        tmp2 = tmp1.generate_random_broadcast_action(world_state)
        #记录本轮action
        new_node_action = NodeAction(_broadcast_blocks_list=tmp2)
        self.add_one_action(new_node_action)
        #记录本轮的action和world state的link
        link_list.append((copy.deepcopy(world_state), new_node_action))
        #self.lastest_action = 1

    def get_lastest_action_flag(self): #返回最新的动作是什么动作，False表示挖矿，True表示广播
        if self.actions[-1].get_hp_rate_list()[0] == 0 and self.actions[-1].get_hp_rate_list()[1] == 0 and self.actions[-1].get_hp_rate_list()[2] == 0: #检测最新动作的前三位是否为0，若是，则表示此动作是一个广播动作
            return True
        else:
            return False

    def get_lastest_broadcast_action_index(self): #获得最新的广播动作的编号：actions[index],若返回1则说明不存在广播动作
        index = -1
        while True:
            if self.actions[index].get_hp_rate_list()[0] == 0 and self.actions[index].get_hp_rate_list()[1] == 0 and self.actions[index].get_hp_rate_list()[2] == 0: #检测最新动作的前三位是否为0，若是，则表示此动作是一个广播动作
                return index
            elif abs(index - 1) > len(self.actions):
                return 1
            else:
                index = index - 1

    def get_lastest_mine_action_index(self): #获得最新的挖矿动作的编号：actions[index],若返回1则说明不存在广播动作
        index = -1
        while True:
            if self.actions[index].get_hp_rate_list()[0] > 0 and self.actions[index].get_hp_rate_list()[1] > 0 and self.actions[index].get_hp_rate_list()[2] > 0: #检测最新动作的前三位是否为0，若是，则表示此动作是一个广播动作
                return index
            elif abs(index - 1) > len(self.actions):
                return 1
            else:
                index = index - 1

    def refresh_action(self):
        pass

    #计算当前钱包的余额并返回，只要最长链改变了都要重新计算钱包余额。
    def compute_wallet(self, world_state): 
        tmp_block = world_state.Global_chain.head
        tmp_wallet = 0
        while True:
            tmp_wallet += int(tmp_block.bit)
            if tmp_block.get_next():
                tmp_block = tmp_block.next[0]
            else:
                break
        tmp_wallet = tmp_wallet * BLOCK_REWARD
        self.set_wallet(tmp_wallet)

    #节点的广播操作，返回一个布尔值，表示是否更新了global chain
    def broadcast(self, world_state): #broadcast_blocks_arr是需要广播的区块的index的列表
        #print('Local node begin broadcast...')
        flag = False
        for i in self.get_broadcast_action():
            block = world_state.Local_chain.get_block_by_index(i)
            if block:
                block.broadcasted = True
                if world_state.refresh_longest_chain():
                    flag = True
        return flag
    
    #节点的待机（无动作）操作
    def do_nothing(self):
        print('Do nothing')

class Game:
    def __init__(self,  _hash_difficulty,  _node_number, _local_node, _other_node, _world_state, _link_list=[]):
        self.round = 0 #挖矿进行的轮次
        self.hash_difficulty = _hash_difficulty #挖矿难度
        self.node_number = _node_number #除本地节点外，参加主游戏的节点数
        self.local_node = _local_node #本地节点
        self.other_node = _other_node #非本地节点（其他节点）
        self.game_round = 0 #GAME进行的轮次数（以一次主链的更新为一个epoch）
        self.world_state = _world_state #世界状态（列表）
        self.world_state_link_node_actions = _link_list #世界状态和动作的对应映射表格

    def print_world_state_link_node_actions(self): # 参数类型：class: WorldState ,NodeAction
        for i, (world_state, node_action) in enumerate(self.world_state_link_node_actions):
            print(f"第{i+1}个世界状态：")
            print("世界状态：")
            print("     Local Chain: ")
            print(world_state.Local_chain.print_chain())
            print("     Global Chain: ")
            print(world_state.Global_chain.print_chain())
            print("节点行为：")
            print("     挖矿行为: ")
            print(node_action.get_mine_father_blocks_list() + node_action.get_hp_rate_list())
            print("     广播行为: ")
            print(node_action.get_broadcast_blocks_list())
            print()

    def get_game_round(self):
        return self.game_round
    def game_round_add_one(self):
        self.game_round += 1

    def get_world_state(self):
        return self.world_state
    def add_one_world_state(self, parm): #添加一个世界状态
        self.world_state.append(parm)
    def delete_lastest_world_state(self): #删除最新的世界状态
        del self.world_state[-1]

    def get_world_state_link_node_actions(self):
        return self.world_state_link_node_actions
    def add_one_world_state_link_node_actions(self, parm):
        self.world_state_link_node_actions.append(parm)
    def delete_lastest_world_state_link_node_actions(self): #删除最新
        del self.world_state_link_node_actions[-1]

    def other_mine(self):
        #通过几何分布获取main game的结果和时长
        def check_samples(samples):
            max_value = np.iinfo(np.int64).max # 获取np.int64的最大值
            for i in range(len(samples)):
                if samples[i] > max_value:
                    samples[i] = max_value
            return samples
        other_hp = (self.other_node.hash_power / self.node_number) # 其他单个节点花费在主游戏上的算力
        samples = np.random.geometric(self.hash_difficulty * (other_hp), size = self.node_number) #表示其他节点的挖矿时长列表
        samples = check_samples(samples)
        other_winner_mine_time = np.min(samples) #其他节点中的最短挖矿时长
        return self.other_node.get_mine_father_block(), other_winner_mine_time

    def local_mine(self):
        def check_samples(samples):
            max_value = np.iinfo(np.int64).max # 获取np.int64的最大值
            for i in range(len(samples)):
                if samples[i] > max_value:
                    samples[i] = max_value
            return samples
        local_hp = self.local_node.get_hash_power()
        #flag_broadcast_action_index = self.local_node.get_lastest_broadcast_action_index()
        flag_mine_action_index = self.local_node.get_lastest_mine_action_index()
        if flag_mine_action_index > 0: #表示没有挖矿动作
            self.local_node.random_set_action_mine(self.world_state[-1], self.get_world_state_link_node_actions())
            local_mine_action = self.local_node.actions[-1].combine_rate_and_father()
        else: #有挖矿动作，并提取出来
            local_mine_action = self.local_node.actions[flag_mine_action_index].combine_rate_and_father()
        local_winner_mine_time = 0 #本地挖矿获胜的最短时长
        local_winner_mine_father = -1 #本地挖矿获胜的块的父块
        for local_father, local_rate in local_mine_action.items():
            sample = np.random.geometric(self.hash_difficulty * (local_rate * local_hp), size = 1) #表示本地节点的挖矿时长
            sample = check_samples(sample)
            if local_winner_mine_time != 0:
                if local_winner_mine_time > sample[0]:
                    local_winner_mine_time = sample[0]
                    local_winner_mine_father = local_father
            else:
                local_winner_mine_time = sample[0]
                local_winner_mine_father = local_father
        return local_winner_mine_father, local_winner_mine_time
    
    #在global chain更新的两个时代的间隙中的内循环本地挖矿
    def local_game(self, remain_time, tmp_other_mine_time, tmp_other_mine_father): #tmp_other_mine_time以备内循环本地节点挖矿失败时，再将other node的块广播出去

        local_mine_father, local_mine_time = self.local_mine()
        remain_time -= local_mine_time
        if remain_time >= 0:
            new_block = Block(bit = True, broadcasted = False, index = self.world_state[-1].Local_chain.get_chain_index(), hash_time = local_mine_time)
            #更新local的状态
            self.add_one_world_state(self.world_state[-1]) #新增新的世界状态
            self.world_state[-1].Local_chain.add_block(new_block, local_mine_father) #将新块添加至本地链
            #更新本地节点的action
            self.local_node.random_set_action_broadcast(self.world_state[-1], self.get_world_state_link_node_actions())
            self.add_one_world_state(self.world_state[-1]) #新增新的临时世界状态(供广播使用的临时状态，若广播成功则更新最新的世界状态，反之则删除此状态)
            flag = self.local_node.broadcast(self.world_state[-1])
            if flag: #本地节点获胜，更新global state
                self.other_node.set_mine_father_block(self.world_state[-1].Global_chain.get_lastest_block().get_index()) #其他节点更新挖矿父区块，注意：不能用new_block
                #print('---Local node win! and wallet is ' + str(self.local_node.get_wallet(self.world_state)) + ' ---')
                self.start_game()
            else: #无人获胜，本地节点继续间隙内的本地挖矿
                #print('and local node do not refresh global longest chain.')
                self.delete_lastest_world_state() #未更新世界状态，固删除临时状态
                self.local_game(remain_time, tmp_other_mine_time, tmp_other_mine_father)
        else: #间隙中的内循环本地挖矿并未成功
            new_block = Block(bit = False, broadcasted = True, index = self.world_state[-1].Local_chain.get_chain_index(), hash_time = tmp_other_mine_time)
            #其他节点获胜，更新global和local的状态
            self.add_one_world_state(self.world_state[-1]) #新增新的世界状态
            self.world_state[-1].Local_chain.add_block(new_block, tmp_other_mine_father) #将新块添加至本地链
            #print('Other node broadcast...')
            if self.world_state[-1].refresh_longest_chain():
                #print('---Other node win!---')
                pass
            #更新所有节点的action
            self.other_node.set_mine_father_block(new_block.get_index())
            if self.get_game_round() != MINE_ROUND: #即，还有下一轮GAME，则需要更新动作
                self.local_node.random_set_action_mine(self.world_state[-1], self.get_world_state_link_node_actions())
            #进入下一轮GAME
            self.start_game()

    def start_game(self):
        
        self.game_round_add_one()
        if self.get_game_round() > MINE_ROUND: #超出最大模拟轮次则退出模拟。
            #print('==========GAME QUIT==========')
            return
        #print('==========GAME ROUND: ' + str(self.get_game_round()) + '==========')

        other_mine_father, other_mine_time = self.other_mine()
        local_mine_father, local_mine_time = self.local_mine()
        #print('OTHER: other_mine_father index = ' + str(other_mine_father) + '; other_mine_time = ' + str(other_mine_time))
        #print('LOCAL: local_mine_father index = ' + str(local_mine_father) + '; local_mine_time = ' + str(local_mine_time))
        if other_mine_time < local_mine_time: # 非本地节点获胜，并且更新global chain
            new_block = Block(bit = False, broadcasted = True, index = self.world_state[-1].Local_chain.get_chain_index(), hash_time = other_mine_time)
            #更新global和local的状态
            self.add_one_world_state(self.world_state[-1]) #新增新的世界状态
            self.world_state[-1].Local_chain.add_block(new_block, other_mine_father) #将新块添加至本地链
            #print('Other node broadcast...')
            if self.world_state[-1].refresh_longest_chain():
                #print('---Other node win!---')
                pass
            #更新节点的action
            self.other_node.set_mine_father_block(new_block.get_index())
            if self.get_game_round() != MINE_ROUND: #即，还有下一轮GAME，则需要更新动作
                self.local_node.random_set_action_mine(self.world_state[-1], self.get_world_state_link_node_actions())
            #进入下一轮GAME
            self.start_game()
        else: #本地节点获胜，更新local chain，但并未广播，所以不更新global chain！！！！
            new_block = Block(bit = True, broadcasted = False, index = self.world_state[-1].Local_chain.get_chain_index(), hash_time = local_mine_time)
            #更新global和local的状态
            self.add_one_world_state(self.world_state[-1]) #新增新的世界状态
            self.world_state[-1].Local_chain.add_block(new_block, local_mine_father) #将新块添加至本地链
            #更新本地节点的action
            self.local_node.random_set_action_broadcast(self.world_state[-1], self.get_world_state_link_node_actions())
            self.add_one_world_state(self.world_state[-1]) #新增新的临时世界状态(供广播使用的临时状态，若广播成功则更新最新的世界状态，反之则删除此状态)
            flag = self.local_node.broadcast(self.world_state[-1])
            if flag: #更新global state，则进入下一轮GAME
                self.other_node.set_mine_father_block(self.world_state[-1].Global_chain.get_lastest_block().get_index()) #其他节点更新挖矿父区块
                #print('---Local node win! and wallet is ' + str(self.local_node.get_wallet(self.world_state)) + ' ---')
                self.start_game()
            else: #不更新global state
                #print('and local node do not refresh global longest chain.')
                self.delete_lastest_world_state() #未更新世界状态，固删除临时状态
                self.local_game(other_mine_time - local_mine_time, other_mine_time, other_mine_father)


if __name__ == "__main__":
    world_state = []
    init_world_state = WorldState() 
    world_state.append(init_world_state) #创建世界状态（本地链和全局链等）
    local_node = Node(global_time = Global_time, hash_power = LOCAL_HASH_POWER, cost = 0, wallet = 0)
    init_link_list = []
    local_node.random_set_action_mine(world_state[0], init_link_list) #随机设置本地节点第一轮挖矿动作
    other_node = OtherNode(global_time = Global_time, hash_power = GLOBAL_HASH_POWER, cost = 0, wallet = 0)
    other_node.set_mine_father_block(0) #设置非本地节点第一轮挖矿动作

    #设置游戏
    new_game = Game(HASH_DIFFICULTY, NODE_NUMBER, local_node, other_node, world_state, init_link_list)
    new_game.start_game()
    
    print('//////////////////Node wallet://////////////////')
    print(str(local_node.get_wallet(new_game.get_world_state()[-1])))
    print('//////////////////Node cost://////////////////')
    print(str(local_node.get_cost()))
    print('//////////////////Local Chain://////////////////')
    new_game.get_world_state()[-1].Local_chain.print_chain()
    print('//////////////////Global Chain://////////////////')
    new_game.get_world_state()[-1].Global_chain.print_chain()
    print('//////////////////Link List://////////////////')
    new_game.print_world_state_link_node_actions()
    print('link list len is ' + str(len(new_game.get_world_state_link_node_actions())))
    '''
    print('//////////////////LSTM INPUT://////////////////')
    print('window size = 5')
    print('Local_chain: ' + str(world_state.Local_chain.chain_4_LSTM_input(5)))
    print('Global_chain: ' + str(world_state.Global_chain.chain_4_LSTM_input(5)))
    print('window size = 10')
    print('Local_chain: ' + str(world_state.Local_chain.chain_4_LSTM_input(10)))
    print('Global_chain: ' + str(world_state.Global_chain.chain_4_LSTM_input(10)))
    print('window size = 15')
    print('Local_chain: ' + str(world_state.Local_chain.chain_4_LSTM_input(15)))
    print('Global_chain: ' + str(world_state.Global_chain.chain_4_LSTM_input(15)))
    print('window size = 20')
    print('Local_chain: ' + str(world_state.Local_chain.chain_4_LSTM_input(20)))
    print('Global_chain: ' + str(world_state.Global_chain.chain_4_LSTM_input(20)))
    
    #设置新的游戏
    new_game2 = Game(HASH_DIFFICULTY, NODE_NUMBER, local_node, other_node, new_game.get_world_state())
    new_game2.start_game()

    print('//////////////////Node wallet://////////////////')
    print(str(local_node.get_wallet(new_game.get_world_state())))
    print('//////////////////Node cost://////////////////')
    print(str(local_node.get_cost()))
    print('//////////////////Local Chain://////////////////')
    world_state.Local_chain.print_chain()
    print('//////////////////Global Chain://////////////////')
    world_state.Global_chain.print_chain()
    
    print('window size = 5')
    print('Local_chain: ' + str(world_state.Local_chain.chain_4_LSTM_input(5)))
    print('Global_chain: ' + str(world_state.Global_chain.chain_4_LSTM_input(5)))
    print('window size = 10')
    print('Local_chain: ' + str(world_state.Local_chain.chain_4_LSTM_input(10)))
    print('Global_chain: ' + str(world_state.Global_chain.chain_4_LSTM_input(10)))
    print('window size = 15')
    print('Local_chain: ' + str(world_state.Local_chain.chain_4_LSTM_input(15)))
    print('Global_chain: ' + str(world_state.Global_chain.chain_4_LSTM_input(15)))
    print('window size = 20')
    print('Local_chain: ' + str(world_state.Local_chain.chain_4_LSTM_input(20)))
    print('Global_chain: ' + str(world_state.Global_chain.chain_4_LSTM_input(20)))
    '''
