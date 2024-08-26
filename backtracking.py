# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/3/30 11:58
@desc: 
"""

def letterCombinations(digits: str):
    """17.digits的所有英文组合"""
    if digits == "":
        return []
    match_dict = {
        '2': ['a', 'b', 'c'],
        '3': ['d', 'e', 'f'],
        '4': ['g', 'h', 'i'],
        '5': ['j', 'k', 'l'],
        '6': ['m', 'n', 'o'],
        '7': ['p', 'q', 'r', 's'],
        '8': ['t', 'u', 'v'],
        '9': ['w', 'x', 'y', 'z']
    }
    def backtrack(c, digits):
        if len(digits) == 0:
            res.append(c)
        else:
            for ch in match_dict[digits[0]]:
                backtrack(c + ch, digits[1:])

    res = []
    backtrack("", digits)
    return res

def combine(n: int, k: int):
    """77.[1, n]中所有k个数的组合"""
    res = []
    def recrusion(l, j):
        if len(l) == k:
            res.append(l)
            return
        for i in range(j, n+1):
            recrusion(l+[i], i+1)
    recrusion([], 1)
    return res

def permute(nums):
    """46.全排列"""
    def backtrack(j, l):
        if len(l) == len(nums):
            res.append(l)
            return
        for i in range(len(nums)):
            if i == j or nums[i] in l:
                continue
            backtrack(i, l+nums[i])
    res = []
    backtrack(len(nums), [])
    return res

def combinationSum(candidates, target: int):
    """组合总数"""
    def backtrack(l, target):
        if target < 0:
            return
        if target == 0:
            l.sort()
            if l not in res:
                res.append(l)
        for i in range(len(candidates)):
            backtrack(l+[candidates[i]], target-candidates[i])
    res = []
    backtrack([], target)
    return res

def generateParenthesis(n: int):
    """22.括号生成"""
    def backtrack(path, left, right):
        if len(path) == 2*n:
            res.append(path)
            return
        if left > n or right > left:
            return
        backtrack(path+"(", left+1, right)
        backtrack(path+")", left, right+1)
    res = []
    backtrack("", 0, 0)
    return res

def subsets(nums):
    """78.所有子集"""
    # 迭代
    res = [[]]
    for num in nums:
        res = res + [[num] + n for n in res]
    return res

    # 回溯：
    def recrusion(j, l):
        res.append(l)
        for i in range(j, len(nums)):
            recrusion(i + 1, l + [nums[i]])

    res = []
    recrusion(0, [])
    return res


def exist(board, word) -> bool:
    """79.单词搜索，搜索是否存在一条路径满足word"""
    def recrusion(i, j, k, visited):
        if k == n:
            return True
        for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            tmp_i, tmp_j = x + i, y + j
            if 0<=tmp_i<row and 0<=tmp_j<col and (tmp_i, tmp_j) not in visited and board[tmp_i][tmp_j] == word[k]:
                visited.add((tmp_i, tmp_j))
                if recrusion(tmp_i, tmp_j, k+1, visited):
                    return True
                visited.remove((tmp_i, tmp_j))
        return False

    n = len(word)
    row, col = len(board), len(board[0])
    for i in range(row):
        for j in range(col):
            if board[i][j] == word[0] and recrusion(i, j, 1, {(i, j)}):
                return True
    return False


def partition(s: str):
    """131.分割回文子串，使得每个子串都是回文子串"""
    def recrusion(s, l):
        if not s:
            res.append(l)
            return
        for i in range(1, len(s)+1):
            if s[:i] == s[:i][::-1]:
                recrusion(s[i:], l+[s[:i]])
    res = []
    recrusion(s, [])
    return res
