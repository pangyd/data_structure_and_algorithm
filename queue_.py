# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/3/24 15:07
@desc:
"""
from collections import deque


def double_queue():
    q = deque([1, 2, 3], maxlen=5)
    q.append(1)
    q.appendleft(1)
    q.pop()
    q.popleft()

    def tail(n):
        """查最后n行数据"""
        with open('a.txt', 'r') as f:
            q = deque(f, n)
            return q

    for line in tail(3):
        print(line, end='')



maze = [[1, 0, 0, 0], [1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 1, 1]]

dirs = [
    lambda x, y: (x-1, y),
    lambda x, y: (x+1, y),
    lambda x, y: (x, y-1),
    lambda x, y: (x, y+1)
]
# 广度优先搜索
def maze_path(x1, y1, x2, y2):
    queue = deque()
    queue.append((x1, y1, -1))
    path = []
    while len(queue) > 0:
        curNode = queue.pop()   # 搜索到该点，并从队列中取出
        path.append(curNode)   # 存放经过的路径!!!
        if curNode[0] == x2 and curNode[1] == y2:
            # 遍历找出所有上一个送来的路径
            curNode = path[-1]
            realpath = []
            while curNode[2] == -1:
                realpath.append(curNode[:2])
                curNode = path[curNode[2]]   # 找到上一个节点
            realpath.append(curNode[:2])   # 放入起点
            realpath.reverse()

        for dir in dirs:
            nextNode = dir(curNode[0], curNode[1])
            if maze[nextNode[0]][nextNode[1]] == 0:
                queue.append((nextNode[0], nextNode[1], len(path) - 1))   # 该位置是由上一个位置送来的
                maze[nextNode[0]][nextNode[1]] = 2   # 记录该点已经走过了
    else:
        return False

