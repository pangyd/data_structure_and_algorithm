# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/4/1 14:23
@desc: 
"""

def func(x):
    import math
    n = int(math.log10(x)) + 1
    return x / (10 ** n - 1)
def func2(x, y):
    if x + y < y + x:
        return 1
    elif x + y > y + x:
        return -1
    else:
        return 0
nums = [12, 65, 9, 82, 5]
# 1.
# nums = sorted(nums, key=func, reverse=True)
# 2.
nums = [str(i) for i in nums]
from functools import cmp_to_key
nums.sort(key=cmp_to_key(func2))
print(nums)
nums = [str(i) for i in nums]
num = "".join(nums)
print(num)


def partitionLabels(s):
    """763.划分字母区间，使得每个区间相互独立"""
    tmp = 0
    x = ""
    y = 0
    res = []
    for i in range(len(s)):
        tmp += 1
        x += s[i]
        for c in set(x):
            if c not in s[i+1]:
                y += 1
            else:
                y = 0
                break
            if y == len(set(x)):
                res.append(tmp)
                tmp = 0
                x = ""
                y = 0
    return res