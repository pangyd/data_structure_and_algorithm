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