# n, m = map(int, input().split())
# a = [int(c) for c in input().split()]

a = (1, 2, 3, 3, 4, 2)
b = {1: 1, 2: 2}
c = {1: 2, 3: 4}
b.update(c)
del b[1]
value = b.pop(2)
print(b)
print(value)

import json
d = json.dumps(b)
print(type(d))
e = json.loads(d)
for key in e.keys():
    print(type(key))
f = eval(d)
print(f)

a = set([1, 1, 2, 3, 4])
a.add(5)
a.remove(5)
a.discard(3)
a.update(set({6}))
print(a)

a = "lidsnfoianv"
print(a[0:10:2])

# a = [int(b) for b in input().split()]
# print(a)
# assert len(a) == 5


import torch.nn as nn
import torch
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.bn1 = nn.BatchNorm2d(num_features=3)
        self.conv2 = nn.Conv2d(3, 3, 1)
    def forward(self, x):
        return self.conv2(self.bn(self.conv1(x)))

model = model()
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        # print(layer.parameters)
        layer.parameters = torch.randn(size=(3, 3, 1, 1))
        # print(layer.parameters)
torch.manual_seed(123)
a = torch.rand(size=(3, 3))
print(a)
b = torch.randn(size=(3, 3))
print(b)
import numpy as np
a = list(map(str, "asd"))
print(a)

a = iter([1, 2, 3])
for _ in range(4):
    print(next(a, "迭代结束"))

import time
now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
nowtime = time.asctime(time.localtime(time.time()))
nowtime = str(nowtime).split(" ")
from datetime import datetime
now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(2>>4)
print(2<<4)

import os
print(os.getcwd())
print(os.environ)
for root, dir, file in os.walk(os.path.dirname(__file__)):
    print(root, dir, file)

import sys
print(sys.path)
from pathlib import Path
print(str(Path.cwd().parent.joinpath("test", "train")))

import platform
print(platform.system(), platform.node(), platform.machine())

import copy
l1 = [1, [2]]
l2 = l1
l3 = copy.copy(l1)
l1[1].append(3)
print(l3)

class A:
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b
    def aaa(self, xx):
        return xx
mod = A()
print(mod.__dir__())

print(eval("1+2"))

def XXX():
    return 2
XXX()

a = [1, 2, 3]
a.remove(1)
del a[0]
print(a)

a = [int(b) for b in input().split()]
print(a)

