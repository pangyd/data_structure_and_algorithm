class NodeList():
    def __init__(self, val=None):
        self.val = val
        self.next = None


a = NodeList(1)
b = NodeList(2)
a.next = b

# 反转k




n = 10
a = []
b = []

a_worker = []
b_worker = []
sum_price = 0

for i in range(2 * n):
    if a[i] <= b[i]:
        if len(a_worker) < n:
            a_worker.append(i)
            sum_price += a[i]
        else:
            b_worker.append(i)
            sum_price += b[i]
    elif a[i] > b[i]:
        if len(b_worker) < n:
            b_worker.append(i)
            sum_price += b[i]
        else:
            a_worker.append(i)
            sum_price += a[i]
