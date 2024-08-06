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

def removeNthFromEnd(head, n: int):
    """19.删除第n个节点"""
    dummy = ListNode(0)
    dummy.next = head  # 将head的头节点给dummy.next

    slow, fast = dummy, dummy
    for _ in range(n):
        fast = fast.next  # 快指针先走n步

    # 同时走，最终slow停在需要删除节点的前一个位置
    while fast and fast.next:
        slow = slow.next
        fast = fast.next

    slow.next = slow.next.next
    return dummy.next  # head的头节点

def mergeTwoLists(list1, list2):
    """21.合并两个链表"""
    if not list1:
        return list2
    elif not list2:
        return list1

    elif list1.val <= list2.val:
        list1.next = mergeTwoLists(list1.next, list2)  # 指向其余节点的合并结果
        return list1
    else:
        list2.next = mergeTwoLists(list1, list2.next)
        return list2