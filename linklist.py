# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/3/25 0:20
@desc: 
"""

class Node:
    def __init__(self, item):
        self.item = item
        self.next = None


def create_linklist_head(l):
    """头插法"""
    head = Node(l[0])
    for element in l[1:]:
        node = Node(element)
        node.next = head
        head = node
    return head

def print_linklist(head):
    while head:
        print(head.item, end=', ')
        head = head.next

def create_linklist_tail(l):
    """尾插法"""
    head = Node(l[0])
    tail = head   # 最开始头尾一样
    for element in l[1:]:
        node = Node(element)
        tail.next = node
        tail = node
    return head

linklist = create_linklist_head([1, 2, 3, 4])

print_linklist(linklist)

class Double_linklist():
    def __init__(self, curNode):
        self.curNode = curNode
    def add(self, p):
        p.next = self.curNode.next
        self.curNode.next.prior = p
        p.prior = self.curNode
        self.curNode.next = p
    def delete(self):
        p = self.curNode.next
        self.curNode.next = p.next
        p.next.prior = self.curNode
        del p
