import numpy as np
import os
import shutil

def copy_files(directory_a, directory_b, directory_c):
    # 遍历读取目录A中的文件
    for filename in os.listdir(directory_a):
        file_path = os.path.join(directory_a, filename)

        # 获取文件名的倒数第17位到倒数第3位组成的字符串s
        s = filename[-21:-6]

        # 在目录B中寻找包含字符串s的文件
        for b_filename in os.listdir(directory_b):
            if s in b_filename:
                b_file_path = os.path.join(directory_b, b_filename)

                # 复制文件到目录C中
                shutil.copy2(b_file_path, directory_c)
                break
directory_a = 'D:/train_data/z6/MCTS_output_new/self_mine'
directory_b = 'D:/train_data/z6/MCTS_output_new/high_value_'
directory_c = 'D:/train_data/z6/self_high/'

for i in range(7):
    tmp_b = directory_b + str(i+1)
    tmp_c = directory_c + str(i+1)
    copy_files(directory_a, tmp_b, tmp_c)


