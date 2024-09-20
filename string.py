def minWindow(self, s: str, t: str) -> str:
    """76.最小覆盖子区域：s覆盖t所有字符的最短字符串"""
    import Counter
    res_left, res_right = -1, len(s)
    left = 0
    dic_s = Counter()
    dic_t = Counter(t)
    for right, c in enumerate(s):
        dic_s[c] += 1
        while dic_s >= dic_t:
            if right - left < res_right - res_left:
                res_left, res_right = left, right
            dic_s[s[left]] -= 1
            left += 1
    return "" if res_left < 0 else s[res_left: res_right + 1]