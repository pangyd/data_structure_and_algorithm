# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/4/26 0:07
@desc: 2n个工人，2个工厂，第i个工人分配到第一个的费用为a[i]  b[i],每个工厂需要n名工人
"""

a = []
b = []
n = 10

c = [a[i] - b[i] for i in range(2*n)]

c_sorted = sorted(range(2*n), key=lambda x: c[x], reverse=False)

allocation = [(c_sorted[i], 1) if i < n else (c_sorted[i], 2) for i in range(2*n)]

