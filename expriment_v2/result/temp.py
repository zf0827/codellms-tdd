import os
import re

# 目标目录路径
target_dir = '/home/yunxiang/work_june/expriment_v2/result'

# 正则表达式匹配exp+数字+字母的模式
pattern = r'^exp([6-9]|10)([a-d])$'

# 遍历目标目录
for item in os.listdir(target_dir):
    item_path = os.path.join(target_dir, item)
    
    # 检查是否是目录且匹配我们的模式
    if os.path.isdir(item_path):
        match = re.match(pattern, item)
        if match:
            # 提取数字和字母
            num = match.group(1)
            letter = match.group(2)
            
            # 计算新的数字
            new_num = int(num) - 5
            
            # 创建新的文件夹名称
            new_name = f'exp{new_num}{letter}'
            new_path = os.path.join(target_dir, new_name)
            
            # 重命名文件夹
            print(f'重命名: {item} -> {new_name}')
            os.rename(item_path, new_path)
