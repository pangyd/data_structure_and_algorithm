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