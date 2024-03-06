#!/usr/bin/env python3

import sys
import math
import random
import numpy as np
import miner
from const import *
import copy
import timeit #监控各环节执行，以优化性能
import cProfile #监控各环节执行，以优化性能
import pickle #Python的pickle模块将数据对象序列化并保存到本地文件中
import datetime #系统时间作为文件名，防止记录时同文件名覆盖

class State(object):
  """
  蒙特卡罗树搜索的游戏状态，记录在某一个Node节点下的状态数据，包含当前的游戏得分、当前的游戏round数、从开始到当前的执行记录。

  需要实现判断当前状态是否达到游戏结束状态，支持从Action集合中随机取出操作。
  """

  def __init__(self):
    self.current_value = 0.0
    # For the first root node, the index is 0 and the game should start from 1
    self.current_round_index = 0
    self.current_expand_tree_deep = 0 #记录当前探索的树深度（default_policy过程）
    #此初始化动作设置会不会导致神经网络输出为0？
    self.cumulative_actions = []
    self.current_world_states = [] #存储当前miner的世界状态
    self.local_miner_node = None #存储当前的本地miner
    self.other_miner_node = None #存储当前的其他miner
    #初始化初始动作行为列表
    tmp_init_action = [0] * (OUTPUT_HP_RATE_LEN + OUTPUT_MINE_BLCOK_LEN + OUTPUT_BROADCAST_LEN)
    self.cumulative_actions = self.cumulative_actions + [tmp_init_action]
    self.link_list = [] #(world state, action)元组的数组，添加代码：miner.py中的：link_list.append((copy.deepcopy(world_state), new_node_action))
  
  def add_one_link_list(self, parm):
    self.link_list.append(parm)
  def set_link_list(self, parm):
    self.link_list = parm
  def get_link_list(self):
    return self.link_list
  def print_list(self):
    for i, (world_state, node_action) in enumerate(self.link_list):
      print(f"第{i+1}个世界状态::::::::")
      print("世界状态::::")
      print(" Local Chain: ")
      print(world_state.Local_chain.print_chain())
      print(" Global Chain: ")
      print(world_state.Global_chain.print_chain())
      print("节点行为::::")
      print(" 挖矿行为: ")
      print(node_action.get_mine_father_blocks_list() + node_action.get_hp_rate_list())
      print(" 广播行为: ")
      print(node_action.get_broadcast_blocks_list())
      print('------------------------------')

  def get_LSTM_data(self, init_mine_rate, init_mine_father): #返回模拟多轮后的结果，结果是由多个元组构成的列表，每个元组由需要embed的数据和不需要embed的数据构成。
    #初始化第一轮的节点动作
    last_action_rate = []
    last_action_not_rate = []
    #初始化输出数据
    output_data = []
    for i, (world_state, node_action) in enumerate(self.link_list):
      if i == 0: #第一轮
        #收集第一轮轮的世界状态
        tmp_Local_chain_index, tmp_Local_chain_faher = world_state.Local_chain.chain_4_LSTM_input()
        tmp_Global_chain_index, tmp_Global_chain_faher = world_state.Global_chain.chain_4_LSTM_input()
        output_data = [(tmp_Local_chain_index + tmp_Local_chain_faher + tmp_Global_chain_index + tmp_Global_chain_faher + [0] *  OUTPUT_BROADCAST_LEN + init_mine_father, init_mine_rate)] #初始化的
        last_action_rate = node_action.get_hp_rate_list()
        last_action_not_rate = node_action.get_broadcast_blocks_list() + node_action.get_mine_father_blocks_list() 
      else: #第二轮及以后
        #收集每轮的世界状态
        tmp_Local_chain_index, tmp_Local_chain_faher = world_state.Local_chain.chain_4_LSTM_input()
        tmp_Global_chain_index, tmp_Global_chain_faher = world_state.Global_chain.chain_4_LSTM_input()
        #收集每轮的节点行为
        tmp_action_rate = node_action.get_hp_rate_list()
        tmp_action_not_rate = node_action.get_broadcast_blocks_list() + node_action.get_mine_father_blocks_list()

        output_data_need_embed = tmp_Local_chain_index + tmp_Local_chain_faher + tmp_Global_chain_index + tmp_Global_chain_faher + last_action_not_rate
        output_data_noneed_embed = last_action_rate
        output_data.append((output_data_need_embed, output_data_noneed_embed)) #添加一对元组

        last_action_not_rate = tmp_action_not_rate
        last_action_rate = tmp_action_rate
      
    return output_data


  def print_last_chain(self):
    print(" Local Chain: ")
    self.current_world_states[-1][-1].Local_chain.print_chain()
    print(" Global Chain: ")
    self.current_world_states[-1][-1].Global_chain.print_chain()

  def add_one_current_world_states(self, parm):
    self.current_world_states.append(parm)
  
  def get_current_world_states(self):
    return self.current_world_states

  def set_local_miner_node(self, parm):
    self.local_miner_node = parm

  def get_local_miner_node(self):
    return self.local_miner_node

  def set_other_miner_node(self, parm):
    self.other_miner_node = parm

  def get_other_miner_node(self):
    return self.other_miner_node

  def get_current_value(self):
    return self.current_value

  def set_current_value(self, value):
    self.current_value = value

  def get_current_round_index(self):
    return self.current_round_index

  def set_current_round_index(self, turn):
    self.current_round_index = turn

  def get_current_expand_tree_deep(self):
    return self.current_expand_tree_deep

  def set_current_expand_tree_deep(self, parm):
    self.current_expand_tree_deep = parm

  def get_cumulative_actions(self):
    return self.cumulative_actions

  def set_cumulative_actions(self, actions):
    self.cumulative_actions = actions

  def is_terminal(self):
    # The round index starts from 1 to max round number
    return self.current_round_index == MAX_ROUND_NUMBER

  def is_default_policy_end(self):
    return self.current_expand_tree_deep >= MAX_ROUND_NUMBER

  def compute_reward(self):
    #修改为符合ALpha miner的奖励计算函数
    return self.get_current_value()

  def get_next_state_with_random_choice(self): #相当于miner中的一轮GAME，即，全局链至少增加一个块
    #flag用于记录是广度探索还是深度探索，因为广度探索树Node不需要继承link-list，而深度探索需要。
    #flag = True表示深度探索，False表示广度探索
    #设置本轮游戏
    init_link_list = []
    one_game = miner.Game(HASH_DIFFICULTY, NODE_NUMBER, 
                      self.get_local_miner_node(), 
                      self.get_other_miner_node(), 
                      self.get_current_world_states()[-1],
                      init_link_list)
    #开始本轮游戏
    one_game.start_game()
    #获取本轮游戏结束后的各个状态
    now_world_state = one_game.get_world_state() #本轮的世界状态(列表)
    now_local_wallet = self.get_local_miner_node().get_wallet(now_world_state[-1]) #本轮的钱包信息
    node_action_flag = self.get_local_miner_node().get_lastest_action_flag()
    node_action = [] #节点本轮的动作，形状：x（x表示本轮local节点的行动次数） * （OUTPUT_MINE_BLCOK_LEN + OUTPUT_HP_RATE_LEN + OUTPUT_BROADCAST_LEN）
    #收集local节点本轮的actions列表（可能是多个动作的集合）
    for i, one_action in enumerate(self.get_local_miner_node().actions):
      if len(node_action) != 0:
        tmp_array = one_action.get_hp_rate_list() + one_action.get_mine_father_blocks_list() + one_action.get_broadcast_blocks_list()
        node_action = np.vstack((node_action, tmp_array))
      else:
        node_action = one_action.get_hp_rate_list() + one_action.get_mine_father_blocks_list() + one_action.get_broadcast_blocks_list()
    #初始化local node.actions
    tmp_init_action = miner.NodeAction()
    self.get_local_miner_node().actions = []
    self.get_local_miner_node().actions.append(tmp_init_action)

    #更新状态
    next_state = State()
    next_state.add_one_current_world_states(now_world_state)
    next_state.set_current_round_index(self.current_round_index + 1)
    next_state.set_current_value(now_local_wallet)
    next_state.set_local_miner_node(self.get_local_miner_node())
    next_state.set_other_miner_node(self.get_other_miner_node())
    next_state.set_cumulative_actions(self.cumulative_actions + [node_action])
    next_state.set_link_list(one_game.get_world_state_link_node_actions())

    return next_state

  def __repr__(self):
    return "State: {}, value: {}, round: {}".format(
        hash(self), self.current_value, self.current_round_index)
    '''
    return "State: {}, value: {}, round: {}, choices: {}".format(
        hash(self), self.current_value, self.current_round_index,
        self.cumulative_actions)
    '''

class Node(object):
  """
  蒙特卡罗树搜索的树结构的Node，包含了父节点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
  """

  def __init__(self):
    self.parent = None
    self.children = []

    self.visit_times = 0
    self.quality_value = 0.0

    self.state = None

  def set_state(self, state):
    self.state = state

  def get_state(self):
    return self.state

  def get_parent(self):
    return self.parent

  def set_parent(self, parent):
    self.parent = parent

  def get_children(self):
    return self.children

  def get_visit_times(self):
    return self.visit_times

  def set_visit_times(self, times):
    self.visit_times = times

  def visit_times_add_one(self):
    self.visit_times += 1

  def get_quality_value(self):
    return self.quality_value

  def set_quality_value(self, value):
    self.quality_value = value

  def quality_value_add_n(self, n):
    self.quality_value += n

  def is_all_expand(self):
    #return len(self.children) == AVAILABLE_CHOICE_NUMBER
    return False

  def add_child(self, sub_node):
    sub_node.set_parent(self)
    self.children.append(sub_node)

  def __repr__(self):
    return "Node: {}, Q/N: {}/{}, state: {}".format(
        hash(self), self.quality_value, self.visit_times, self.state)

def tree_policy(node):
  """
  蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。

  基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
  """
  # Check if the current node is the leaf node

  if node.is_all_expand():
    node = best_child(node, True)
  else:
    # Return the new sub node
    sub_node = expand(node)
    return sub_node

  return node
  '''
  while node.get_state().is_terminal() == False:
    #print('tree_policy round = ' + str(node.get_state().get_current_round_index()))
    if node.is_all_expand():
      node = best_child(node, True)
    else:
      # Return the new sub node
      sub_node = expand(node)
      return sub_node

  # Return the leaf node
  return node
  '''

def default_policy(node):
  """
  蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward。注意输入的节点应该不是子节点，而且是有未执行的Action可以expend的。

  基本策略是随机选择Action。
  """
  # Get the state of the game
  current_state = copy.deepcopy(node.get_state()) #深拷贝此节点状态，防止深度随机探索时修改父节点的state。
  for _ in range(MAX_ROUND_NUMBER):
    # Pick one random action to play and get next state
    current_state = current_state.get_next_state_with_random_choice()
  '''
  # Run until the game over
  while current_state.is_default_policy_end() == False:
    #print('default_policy round = ' + str(current_state.get_current_round_index()))
    # Pick one random action to play and get next state
    current_state = current_state.get_next_state_with_random_choice()
  '''
  final_state_reward = current_state.compute_reward()
  return final_state_reward

def expand(node):
  """
  输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
  """
  tried_sub_node_states = [
      sub_node.get_state() for sub_node in node.get_children()
  ]
  now_state = copy.deepcopy(node.get_state()) #当前状态的深拷贝
  new_state = now_state.get_next_state_with_random_choice()

  # Check until get the new state which has the different action from others
  while new_state in tried_sub_node_states:
    new_state = node.get_state().get_next_state_with_random_choice()

  sub_node = Node()
  sub_node.set_state(new_state)
  node.add_child(sub_node)

  return sub_node

def best_child(node, is_exploration):
  """
  使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
  """
  #print('进入 best_child 函数')
  # TODO: Use the min float value
  best_score = -sys.maxsize
  best_sub_node = None

  # Travel all sub nodes to find the best one
  for sub_node in node.get_children():

    # Ignore exploration for inference
    if is_exploration:
      C = 1 / math.sqrt(2.0)
    else:
      C = 0.0

    # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
    left = sub_node.get_quality_value() / sub_node.get_visit_times()
    right = 2.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
    score = left + C * math.sqrt(right)

    if score > best_score:
      best_sub_node = sub_node
      best_score = score
  #print('退出 best_child 函数')
  return best_sub_node

def backup(node, reward):
  """
  蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
  """
  # Update util the root node
  while node != None:
    # Update the visit times
    node.visit_times_add_one()

    # Update the quality value
    node.quality_value_add_n(reward)

    # Change the node to the parent node
    node = node.parent

def monte_carlo_tree_search(node):
  """
  实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，然后返回只要exploitation最高的子节点。

  蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
  前两步使用tree policy找到值得探索的节点。
  第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
  最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。

  进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
  """
  # computation_budget
  computation_budget = COMPUTATION_BUDGET

  # Run as much as possible under the computation budget
  for i in range(computation_budget):

    # 1. Find the best node to expand
    expand_node = tree_policy(node)

    # 2. Random run to add node and get reward
    reward = default_policy(expand_node)

    # 3. Update all passing nodes with reward
    backup(expand_node, reward)

  # N. Get the best next node
  best_next_node = best_child(node, False)
  best_next_node.get_state().set_link_list(node.get_state().get_link_list() + best_next_node.get_state().get_link_list())
  return best_next_node

def print_tree(node, indent=0):
    print(str(indent) + "  " * indent + "- Node: {}, Q/N: {}/{}, state: {}".format(
        hash(node), node.quality_value, node.visit_times, node.state))
    for child in node.children:
        print_tree(child, indent + 1)

def main(standard_line, simulation_round):
  # Create the initialized state and initialized node
  init_state = State()
  #初始化STATE所需的各种参数
  world_state = []
  init_world_state = miner.WorldState() 
  world_state.append(init_world_state) #创建世界状态（本地链和全局链等）
  local_node = miner.Node(global_time = miner.Global_time, hash_power = LOCAL_HASH_POWER, cost = 0, wallet = 0)
  init_link_list = []
  local_node.random_set_action_mine(world_state[0], init_link_list) #随机设置本地节点第一轮挖矿动作
  other_node = miner.OtherNode(global_time = miner.Global_time, hash_power = GLOBAL_HASH_POWER, cost = 0, wallet = 0)
  other_node.set_mine_father_block(0) #设置非本地节点第一轮挖矿动作
  #初始化init_state的各项参数
  init_state.add_one_current_world_states(world_state)
  init_state.set_local_miner_node(local_node)
  init_state.set_other_miner_node(other_node)

  init_node = Node()
  init_node.set_state(init_state)
  current_node = init_node
  last_value = 0 #初始化“上轮奖励值”为0，此值记录每轮local node获得了多少奖励
  init_mine_rate = local_node.get_mine_rate_action() #节点第一轮的挖矿rate列表，用于作为get_LSTM_data函数的输入。
  init_mine_father = local_node.get_mine_father_action() #节点第一轮的挖矿father列表，用于作为get_LSTM_data函数的输入。

  flag_2_value_1_round = 0 #初始化“自私挖矿”的标记
    # Set the rounds to play
  for i in range(GAME_MAX_ROUND):
    print("////////////////Play round: {}////////////////".format(i + 1))
    current_node = monte_carlo_tree_search(current_node)
    print("////////////////Choose node: {}////////////////".format(current_node))
    gap_value = current_node.get_state().get_current_value() - last_value
    if gap_value > 1: #本轮获得的奖励与上轮的差值超过了1，至少为2
      flag_2_value_1_round += gap_value
    last_value = current_node.get_state().get_current_value()
  print("////////////////Best node's chain////////////////")
  current_node.get_state().print_last_chain()

  output_LSTM_data = current_node.get_state().get_LSTM_data(init_mine_rate, init_mine_father)


  #with open('MCTS_output_test/output_LSTM_data_test.pkl', 'wb') as file:
    #pickle.dump(output_LSTM_data, file)

  if current_node.get_state().get_current_value() >= standard_line and current_node.get_state().get_current_value() < standard_line+1:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    if flag_2_value_1_round < 3:
      file_name = 'MCTS_output_new/high_value_1/self_mine_less_3/output_LSTM_data_high_value_{}_{}.pkl'.format(
        simulation_round, current_time)
      with open(file_name, 'wb') as file:
        pickle.dump(output_LSTM_data, file)
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      file_name = 'MCTS_output_new/high_value_1/self_mine_3_4/output_LSTM_data_high_value_{}_{}.pkl'.format(simulation_round, current_time)
      with open(file_name, 'wb') as file:
        pickle.dump(output_LSTM_data, file)
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      file_name = 'MCTS_output_new/high_value_1/self_mine_5_6/output_LSTM_data_high_value_{}_{}.pkl'.format(simulation_round, current_time)
      with open(file_name, 'wb') as file:
        pickle.dump(output_LSTM_data, file)
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      file_name = 'MCTS_output_new/high_value_1/self_mine_7_8/output_LSTM_data_high_value_{}_{}.pkl'.format(simulation_round, current_time)
      with open(file_name, 'wb') as file:
        pickle.dump(output_LSTM_data, file)
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      file_name = 'MCTS_output_new/high_value_1/self_mine_9_10/output_LSTM_data_high_value_{}_{}.pkl'.format(simulation_round, current_time)
      with open(file_name, 'wb') as file:
        pickle.dump(output_LSTM_data, file)
    if flag_2_value_1_round >= 10:
      file_name = 'MCTS_output_new/high_value_1/self_mine_more_10/output_LSTM_data_high_value_{}_{}.pkl'.format(simulation_round, current_time)
      with open(file_name, 'wb') as file:
        pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+1 and current_node.get_state().get_current_value() < standard_line+2:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    self_mine_file_name = ''
    if flag_2_value_1_round < 3:
      self_mine_file_name = 'self_mine_less_3/'
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      self_mine_file_name = 'self_mine_3_4/'
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      self_mine_file_name = 'self_mine_5_6/'
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      self_mine_file_name = 'self_mine_7_8/'
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      self_mine_file_name = 'self_mine_9_10/'
    if flag_2_value_1_round >= 10:
      self_mine_file_name = 'self_mine_more_10/'
    file_name = 'MCTS_output_new/high_value_2/'+self_mine_file_name+'output_LSTM_data_high_value_{}_{}.pkl'.format(simulation_round,
                                                                                            current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+2 and current_node.get_state().get_current_value() < standard_line+3:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    self_mine_file_name = ''
    if flag_2_value_1_round < 3:
      self_mine_file_name = 'self_mine_less_3/'
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      self_mine_file_name = 'self_mine_3_4/'
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      self_mine_file_name = 'self_mine_5_6/'
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      self_mine_file_name = 'self_mine_7_8/'
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      self_mine_file_name = 'self_mine_9_10/'
    if flag_2_value_1_round >= 10:
      self_mine_file_name = 'self_mine_more_10/'
    file_name = 'MCTS_output_new/high_value_3/' + self_mine_file_name + 'output_LSTM_data_high_value_{}_{}.pkl'.format(
      simulation_round,
      current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+3 and current_node.get_state().get_current_value() < standard_line+4:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    self_mine_file_name = ''
    if flag_2_value_1_round < 3:
      self_mine_file_name = 'self_mine_less_3/'
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      self_mine_file_name = 'self_mine_3_4/'
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      self_mine_file_name = 'self_mine_5_6/'
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      self_mine_file_name = 'self_mine_7_8/'
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      self_mine_file_name = 'self_mine_9_10/'
    if flag_2_value_1_round >= 10:
      self_mine_file_name = 'self_mine_more_10/'
    file_name = 'MCTS_output_new/high_value_4/' + self_mine_file_name + 'output_LSTM_data_high_value_{}_{}.pkl'.format(
      simulation_round,
      current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+4 and current_node.get_state().get_current_value() < standard_line+5:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    self_mine_file_name = ''
    if flag_2_value_1_round < 3:
      self_mine_file_name = 'self_mine_less_3/'
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      self_mine_file_name = 'self_mine_3_4/'
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      self_mine_file_name = 'self_mine_5_6/'
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      self_mine_file_name = 'self_mine_7_8/'
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      self_mine_file_name = 'self_mine_9_10/'
    if flag_2_value_1_round >= 10:
      self_mine_file_name = 'self_mine_more_10/'
    file_name = 'MCTS_output_new/high_value_5/' + self_mine_file_name + 'output_LSTM_data_high_value_{}_{}.pkl'.format(
      simulation_round,
      current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+5 and current_node.get_state().get_current_value() < standard_line+6:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    self_mine_file_name = ''
    if flag_2_value_1_round < 3:
      self_mine_file_name = 'self_mine_less_3/'
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      self_mine_file_name = 'self_mine_3_4/'
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      self_mine_file_name = 'self_mine_5_6/'
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      self_mine_file_name = 'self_mine_7_8/'
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      self_mine_file_name = 'self_mine_9_10/'
    if flag_2_value_1_round >= 10:
      self_mine_file_name = 'self_mine_more_10/'
    file_name = 'MCTS_output_new/high_value_6/' + self_mine_file_name + 'output_LSTM_data_high_value_{}_{}.pkl'.format(
      simulation_round,
      current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+6 and current_node.get_state().get_current_value() < standard_line+7:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    self_mine_file_name = ''
    if flag_2_value_1_round < 3:
      self_mine_file_name = 'self_mine_less_3/'
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      self_mine_file_name = 'self_mine_3_4/'
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      self_mine_file_name = 'self_mine_5_6/'
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      self_mine_file_name = 'self_mine_7_8/'
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      self_mine_file_name = 'self_mine_9_10/'
    if flag_2_value_1_round >= 10:
      self_mine_file_name = 'self_mine_more_10/'
    file_name = 'MCTS_output_new/high_value_7/' + self_mine_file_name + 'output_LSTM_data_high_value_{}_{}.pkl'.format(
      simulation_round,
      current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+7 and current_node.get_state().get_current_value() < standard_line+8:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    self_mine_file_name = ''
    if flag_2_value_1_round < 3:
      self_mine_file_name = 'self_mine_less_3/'
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      self_mine_file_name = 'self_mine_3_4/'
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      self_mine_file_name = 'self_mine_5_6/'
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      self_mine_file_name = 'self_mine_7_8/'
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      self_mine_file_name = 'self_mine_9_10/'
    if flag_2_value_1_round >= 10:
      self_mine_file_name = 'self_mine_more_10/'
    file_name = 'MCTS_output_new/high_value_8/' + self_mine_file_name + 'output_LSTM_data_high_value_{}_{}.pkl'.format(
      simulation_round,
      current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+8 and current_node.get_state().get_current_value() < standard_line+9:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    self_mine_file_name = ''
    if flag_2_value_1_round < 3:
      self_mine_file_name = 'self_mine_less_3/'
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      self_mine_file_name = 'self_mine_3_4/'
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      self_mine_file_name = 'self_mine_5_6/'
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      self_mine_file_name = 'self_mine_7_8/'
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      self_mine_file_name = 'self_mine_9_10/'
    if flag_2_value_1_round >= 10:
      self_mine_file_name = 'self_mine_more_10/'
    file_name = 'MCTS_output_new/high_value_9/' + self_mine_file_name + 'output_LSTM_data_high_value_{}_{}.pkl'.format(
      simulation_round,
      current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+9 and current_node.get_state().get_current_value() < standard_line+10:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    self_mine_file_name = ''
    if flag_2_value_1_round < 3:
      self_mine_file_name = 'self_mine_less_3/'
    if flag_2_value_1_round >= 3 and flag_2_value_1_round < 5:
      self_mine_file_name = 'self_mine_3_4/'
    if flag_2_value_1_round >= 5 and flag_2_value_1_round < 7:
      self_mine_file_name = 'self_mine_5_6/'
    if flag_2_value_1_round >= 7 and flag_2_value_1_round < 9:
      self_mine_file_name = 'self_mine_7_8/'
    if flag_2_value_1_round >= 9 and flag_2_value_1_round < 10:
      self_mine_file_name = 'self_mine_9_10/'
    if flag_2_value_1_round >= 10:
      self_mine_file_name = 'self_mine_more_10/'
    file_name = 'MCTS_output_new/high_value_10/' + self_mine_file_name + 'output_LSTM_data_high_value_{}_{}.pkl'.format(
      simulation_round,
      current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)
  elif current_node.get_state().get_current_value() >= standard_line+10:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间
    file_name = 'MCTS_output_new/high_value_more/output_LSTM_data_high_value_{}_{}.pkl'.format(simulation_round, current_time)
    with open(file_name, 'wb') as file:
      pickle.dump(output_LSTM_data, file)


  #print("////////////////Link List////////////////") #打印本次模拟的最优路径的所有状态和动作的组合列表
  #current_node.get_state().print_list()
  #print('link list len is ' + str(len(current_node.get_state().get_link_list())))
  # Print the tree
  #print("Tree structure:")
  #print_tree(init_node)

def get_standard_line(round = GAME_MAX_ROUND):
  #获取local node正常挖矿时的标准线
  #目前只是简单地用概率计算，并没有真正的模拟。
  return LOCAL_HASH_POWER * round * BLOCK_REWARD / (LOCAL_HASH_POWER + GLOBAL_HASH_POWER) + 1

if __name__ == "__main__":
  standard_line = get_standard_line()
  for i in range(SIMULATION_ROUND):
    # 运行性能分析
    main(standard_line, i)