def merge(intervals):
    """56. 插入区间"""
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    res = [intervals[0]]
    for i in range(1, len(intervals)):

        if intervals[i][0] <= intervals[i - 1][1]:
            if intervals[i - 1][1] >= intervals[i][1]:
                intervals[i] = intervals[i - 1]
            else:
                intervals[i] = [intervals[i - 1][0], intervals[i][1]]
                res[-1] = intervals[i]
        else:
            res.append(intervals[i])
    return res

def insert(intervals, newInterval):
    """57.插入区间"""
    if not intervals:
        return [newInterval]
    if newInterval[1] < intervals[0][0]:
        return [newInterval] + intervals
    if newInterval[0] > intervals[-1][-1]:
        return intervals + [newInterval]

    i = 0
    while i < len(intervals) and newInterval[0] > intervals[i][1]:
        i += 1
    left = min(newInterval[0], intervals[i][0])
    tmp = i
    if newInterval[1] < intervals[i][0]:
        return intervals[:i] + [newInterval] + intervals[i:]

    right = newInterval[1]
    while i < len(intervals) and right >= intervals[i][0]:
        right = max(newInterval[1], intervals[i][1])
        i += 1
    return intervals[:tmp] + [[left, right]] + intervals[i:]