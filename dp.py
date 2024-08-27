# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/7/22 21:15
@desc: 
"""
class dp_in_array:
    def canPartition(self, nums) -> bool:
        """416.分割等和子集:nums能不能分割成两个子集，且子集的和相等"""
        if sum(nums) % 2 != 0:
            return False
        half = sum(nums) // 2
        n = len(nums)
        dp = [[False] * (half + 1) for _ in range(n + 1)]
        dp[0][0] = True
        for i in range(1, n + 1):
            for j in range(half + 1):
                if j < nums[i - 1]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
        return dp[n][half]

    def coinChange(self, coins, amount) -> int:
        """凑成硬币总额的最少硬币数"""
        # dp[i][j]:前i个硬币组成j的硬币数
        dp = [[amount+10] * (amount+1) for _ in range(len(coins)+1)]
        for i in range(len(coins)+1):
            dp[i][0] = 0

        for i in range(1, len(coins)+1):
            for j in range(amount+1):
                if j < coins[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i-1]]+1)
        ans = dp[amount][len(coins)]
        return ans if ans != amount+10 else -1

class dp_in_bitree:
    def numTrees(self, n: int) -> int:
        """96.n个节点一个可以组成多少种不同的二叉搜索树"""
        dp = [1, 1]
        if n <= 1:
            return dp[n]
        for m in range(2, n+1):
            s = m - 1
            count = 0
            for i in range(m):
                count += dp[i] * dp[s-i]
            dp[m] = count
        return dp[n]

    def generateTrees(self, n: int):
        """95.n个节点组成二叉搜索树"""
        def bitree(left, right):
            res = []
            if left > right:
                return [None]
            for i in range(left, right + 1):
                l_tree = bitree(left, i - 1)
                r_tree = bitree(i + 1, right)
                for l in l_tree:
                    for r in r_tree:
                        node = TreeNode(i)
                        node.left = l
                        node.right = r
                        res.append(node)
            return res

        res = bitree(1, n)
        return res

    def rob(self, root) -> int:
        """337.打家劫舍3：打劫了当前节点，则不能打劫左右子节点，求最大"""
        def bitree(root):
            if not root:
                return 0, 0
            left = bitree(root.left)
            right = bitree(root.right)
            v1 = root.val + left[1] + right[1]
            v2 = max(left) + max(right)
            return v1, v2
        return max(bitree(root))

class dp_in_string:
    def longestPalindrome(self, s: str) -> str:
        """最长回文子串"""
        n = len(s)
        if n < 2:
            return s
        dp = [[False] * n for _ in range(n)]
        begin = 0
        max_len = 1
        for i in range(n):
            dp[i][i] = True
        for L in range(2, n+1):
            for i in range(n):
                j = L + i - 1
                if j >= n:
                    break
                if j - i < 3:
                    dp[i][j] = True
                else:
                    if s[i] == s[j]:
                        dp[i][j] = dp[i+1][j-1]
                    if dp[i][j] and j - i > max_len:
                        max_len = j - 1
                        begin = i
        return s[begin: begin+max_len]

    def wordBreak(self, s: str, wordDict) -> bool:
        """139.单词拆分"""
        # \  l  e  e  t  c  o  d  e
        # T  F  F  F  T  F  F  F  T
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(n):
            for j in range(i+1, n+1):
                if dp[i] and s[i:j] in wordDict:
                    dp[j] = True
        return dp[-1]

    def minDistance(self, word1: str, word2: str) -> int:
        """72.word1转换成word2的最少操作数"""
        m, n = len(word1), len(word2)
        dp = [[0] * (m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            dp[i][0] = dp[i-1][0] + 1
        for j in range(1, m+1):
            dp[0][j] = dp[0][j-1] + 1

        for i in range(1, n+1):
            for j in range(1, m+1):
                if word1[i] == word2[j]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[n][m]

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        """97.交错字符串"""
        if len(s1) + len(s2) != len(s3):
            return False
        n1, n2 = len(s1), len(s2)
        dp = [[False] * (n2 + 1) for _ in range(n1 + 1)]
        # dp[i][j]: s1的前i个和s2的前i个能否组成s3的前i+j个
        dp[0][0] = True
        for i in range(1, n1 + 1):
            if s1[i - 1] == s3[i - 1] and dp[i - 1][0]:
                dp[i][0] = True
        for j in range(1, n2 + 1):
            if s2[j - 1] == s3[j - 1] and dp[0][j - 1]:
                dp[0][j] = True

        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if (s1[i - 1] == s3[i + j - 1] and dp[i - 1][j]) or (s2[j - 1] == s3[i + j - 1] and dp[i][j - 1]):
                    dp[i][j] = True
        return dp[n1][n2]

class dp_in_bag:
    def numSquares(self, n: int) -> int:
        """279.拆分成完全平方数的最少数量"""
        from collections import deque
        queue = deque()
        queue.append((n, 0))
        while queue:
            num, step = queue.popleft()
            nums = [num-i*i for i in range(1, int(num**0.5)+1)]
            for target in nums:
                if target == 0:
                    return step + 1
                queue.append((target, step+1))
        return 0

class dp_in_sequence:
    def lengthOfLIS(self, nums) -> int:
        n = len(nums)
        if n <= 1:
            return 1
        dp = [1] * n
        res = 1
        max_len = 0
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        res += 1
            max_len = max(max_len, res)
            res = 1
        return max_len
    def findNumberOfLIS(self, nums) -> int:
        """673.求出所有最长连续子序列"""
        n = len(nums)
        if n <= 1:
            return 1
        dp = [1] * n
        counts = [1] * n
        max_len = 0
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        counts[i] = counts[j]
                    elif dp[j] + 1 == dp[i]:
                        counts[i] += counts[j]
            max_len = max(max_len, dp[i])
        res = 0
        for i in range(n):
            if dp[i] == max_len:
                res += counts[i]
        return res

def jump(nums) -> int:
    """45.跳跃到最后一个最少的步数"""
    n = len(nums)
    dp = [float("inf")] * n
    dp[0] = 0
    for i in range(0, n-1):
        if nums[i] == 0:
            continue
        if i + nums[i] + 1 > n:
            for j in range(i+1, n):
                dp[j] = min(dp[i]+1, dp[j])
        else:
            for j in range(i+1, i+nums[i]+1):
                dp[j] = min(dp[i]+1, dp[j])
    return dp[-1]

def climbStairs(n: int) -> int:
    """70.爬楼梯"""
    """f(n)=f(n-1)+f(n-2)"""
    f = [1, 1]
    for i in range(2, n + 1):
        f.append(f[i - 1] + f[i - 2])
    return f[-1]


def maxProduct(nums) -> int:
    """152.乘积最大的子数组"""
    res, pre_max, pre_min = nums[0], nums[0], nums[0]
    for num in nums[1:]:
        cur_max = max(num, num*pre_max, num*pre_min)
        cur_min = min(num, num*pre_min, num*pre_max)
        res = max(res, cur_max)
        pre_max = cur_max
        pre_min = cur_min
    return res


def longestValidParentheses(s: str) -> int:
    """32.最长有效括号数"""
    res = 0
    stack = []
    for i in range(len(s)):
        if not stack or s[i] == '(' or s[stack[-1]] == ')':
            stack.append(i)
        else:
            stack.pop()
            res = max(res, i - (stack[-1] if stack else -1))
    return res