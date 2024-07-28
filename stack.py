# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/3/24 15:36
@desc: 深度优先搜索  --  回溯法
"""
import numpy as np

maze = [[], []]

dirs = [
    lambda x, y: (x-1, y),
    lambda x, y: (x+1, y),
    lambda x, y: (x, y-1),
    lambda x, y: (x, y+1)
]

def maze_path(x1, y1, x2, y2):
    """起点和终点的坐标"""
    stack = []
    stack.append((x1, y1))
    while len(stack) > 0:
        curNode = stack[-1]   # 当前位置
        if curNode == (x2, y2):
            # 打印路线
            for p in stack:
                print(p)
        # 搜索四个方向：上下左右
        for dir in dirs:
            nextNode = dir(curNode[0], curNode[1])   # 下一个位置
            if maze[nextNode[0]][nextNode[1]] == 0:   # 下一点可走
                stack.append(nextNode)
                maze[nextNode[0]][nextNode[1]] = 2
                break
            else:
                maze[nextNode[0]][nextNode[1]] = 2   # 保存已经走过的位置，避免重复走
                stack.pop()
    else:
        return False


