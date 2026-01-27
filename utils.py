'''
    此处用于定义一些全局可用的函数
'''
import pygame
import os
import re

def load_image(image_path):
    try:
        return pygame.image.load(image_path)
    except:
        print(f"Warning: Cannot load image: {image_path}")
        return None
    
def get_next_filename(folder_path, prefix, extension='.txt'):
    """
    获取文件夹中最小的未被使用的文件名
    示例： next_filename = get_next_filename('Map/saved', 'map', '.txt')
          如果Map/saved文件夹中已有map0.txt, map1.txt, map2.txt, 则next_filename为map3.txt
          注意不包含路径
    """
    # 确保文件夹路径存在
    os.makedirs(folder_path, exist_ok=True)
    
    # 获取所有文件
    files = [f for f in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, f))]
    
    # 提取已存在的编号
    existing_numbers = []
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+){re.escape(extension)}$')
    
    for file in files:
        match = pattern.match(file)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    # 找到最小可用的编号
    if not existing_numbers:
        next_num = 0
    else:
        existing_numbers.sort()
        # 寻找缺失的编号
        for i in range(len(existing_numbers)):
            if i != existing_numbers[i]:
                next_num = i
                break
        else:
            next_num = existing_numbers[-1] + 1
    
    # 返回完整路径
    return f"{prefix}{next_num}{extension}"