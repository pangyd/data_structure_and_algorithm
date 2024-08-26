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


def isValid(s: str) -> bool:
    """20.有效的括号"""
    if len(s) == 1:
        return False
    match_dict = {'(': ')', '[': ']', '{': '}'}
    listNode = []
    for char in s:
        if char in match_dict.keys():
            listNode.append(char)
        else:
            if len(listNode) == 0:   # 当前为空栈
                return False
            # 当char in [), ], }]时，判断链表最后一个是否与之匹配，匹配则出栈，不匹配则返回False
            if char == match_dict[listNode[-1]]:
                # 匹配，出栈
                listNode.pop()
            else:
                # 不匹配，返回False
                return False
    # 最后判断栈是否为空
    if len(listNode) == 0:
        return True
    else:
        return False

def generateParenthesis(n: int):
    """22.括号生成"""
    res = []
    def dfs(paths, left, right):
        if left > n or right > left:  # 左括号＞n或右括号＞左括号
            return
        if len(paths) == n * 2:  # 括号成对存在
            res.append(paths)
            return  # 返回后继续递归

        dfs(paths + "(", left + 1, right)
        dfs(paths + ")", left, right + 1)

    dfs('', 0, 0)
    return res

def calculate(s: str) -> int:
    """224.计算器"""
    res, num, sign = 0, 0, 1
    stack = []
    for c in s:
        if c.isdigit():
            num = 10 * num + int(c)
        elif c == "+" or c == "-":
            res += num * sign
            num = 0
            sign = 1 if c == "+" else -1
        elif c == "(":
            stack.append(sign)
            stack.append(res)
            res = 0
            sign = 1
        elif c == ")":
            res += num * sign
            num = 0
            res *= stack.pop()
            res += stack.pop()
    res += num * sign
    return res


def dailyTemperatures(temperatures):
    """739.下一个更高温度是几天之后"""
    stack = []
    res = [0] * len(temperatures)
    for i in range(len(temperatures)):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            x = stack.pop()
            res[x] = i - x
        stack.append(i)
    return res


def decodeString(s: str) -> str:
    """394.字符串解码"""
    res, num = "", ""
    stack = []
    for c in s:
        if "0" <= c <= "9":
            num += c
        elif c == "[":
            stack.append((res, num))
            num = ""
            res = ""
        elif c == "]":
            tmp, k = stack.pop()
            res = tmp + int(k) * res
        else:
            res += c
    return res


def largestRectangleArea(heights) -> int:
    """84.柱状图中最大的矩形"""
    res = 0
    heights = [0] + heights + [0]
    stack = []
    for i in range(len(heights)):
        while stack and heights[stack[-1]] > heights[i]:
            x = stack.pop()
            res = max(res, heights[x] * (i - stack[-1] - 1))
        stack.append(i)
    return res