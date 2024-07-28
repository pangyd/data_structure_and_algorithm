# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/4/7 15:13
@desc: 正序反序相同  --  动态规划
"""

s = 'ibukasbc'

def longestPalindrome(s: str) -> str:
    n = len(s)
    if n == 1:
        return s

    dp = [[False] * n for _ in range(n)]
    max_len = 1
    begin = 0
    for i in range(n):
        dp[i][i] = True
    for L in range(2, n+1):
        for i in range(n):
            j = L + i - 1
            if j >= n:
                break
            if s[i] != s[j]:
                dp[i][j] = False
            else:
                if j - i < 3:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i+1][j-1]
            if dp[i][j] and (L > max_len):
                max_len = L
                begin = i
    return s[begin: begin + max_len]

print(longestPalindrome("abccbdsjnoiion"))