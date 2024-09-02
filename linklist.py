# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/3/25 0:20
@desc: 
"""

class ListNode:
    def __init__(self, item=0, next=None):
        self.item = item
        self.next = next


def create_linklist_head(l):
    """头插法"""
    head = ListNode(l[0])
    for element in l[1:]:
        node = ListNode(element)
        node.next = head
        head = node
    return head

def print_linklist(head):
    while head:
        print(head.item, end=', ')
        head = head.next

def create_linklist_tail(l):
    """尾插法"""
    head = ListNode(l[0])
    tail = head   # 最开始头尾一样
    for element in l[1:]:
        node = ListNode(element)
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


def deleteDuplicates(head):
    """82.删除重复元素"""
    p = dummp = ListNode(next=head)
    while p.next and p.next.next:
        val = p.next.val
        if p.next.next.val == val:
            while p.next and p.next.val == val:
                p.next = p.next.next
        else:
            p = p.next
    return dummp.next

def sortList(self, head):
    """148.排序链表  --  分支（分成两部分）"""
    if not head or not head.next:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next

    slow.next, mid = None, slow.next

    left = self.sortList(head)
    right = self.sortList(mid)

    p = dummy = ListNode()
    while left and right:
        if left.val < right.val:
            p.next = left
            left = left.next
        else:
            p.next = right
            right = right.next
        p = p.next
    p.next = left if left else right
    return dummy.next


def getIntersectionNode(self, headA: ListNode, headB: ListNode):
    """160.相交链表"""
    A, B = headA, headB
    while A != B:
        A = A.next if A else headB
        B = B.next if B else headA
    return A


def swapPairs(head):
    """24.两两交换节点"""
    p = dummy = ListNode()
    dummy.next = head
    while p.next and p.next.next:
        a = p.next
        b = p.next.next
        a.next = b.next
        b.next = a
        p.next = b
        p = p.next.next
    return dummy.next