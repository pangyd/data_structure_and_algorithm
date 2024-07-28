# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/4/7 16:51
@desc: 求最大矩形面积: 遍历找到下降的索引，往回遍历到上一个下降的点，求此时最大面积
"""

heights = [2, 1, 5, 6, 2, 3]

# 一维数组
def matrixArea(heights):
    res = 0
    heights = [0] + heights + [0]
    stack = []
    for i in range(len(heights)):
        while stack and heights[stack[-1]] > heights[i]:   # 找到下降的索引
            tmp = stack.pop()   # 当前最高矩形的索引
            res = max(res, (i - stack[-1] - 1) * heights[tmp])
        stack.append(i)   # 栈中为上升的索引
    return res


# 二维数组，只有'1'和'0'
def matrixArea2(matrix):
    res = 0
    m, n = len(matrix), len(matrix[0])
    pre = [0] * (n+1)

    for i in range(m):
        for j in range(n):
            pre[j] = pre[j] + 1 if matrix[i][j] == '1' else 0

        stack = [-1]
        for k, num in enumerate(pre):
            while stack and pre[stack[-1]] > num:
                tmp = stack.pop()
                res = max(res, pre[tmp] * (k- stack[-1] - 1))
            stack.append(k)
    return res