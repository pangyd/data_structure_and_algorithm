# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/7/22 21:15
@desc: 
"""
# 416.分割等和子集
class Solution:
    def canPartition(self, nums) -> bool:
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

# 凑成硬币总额的最少硬币数
class Solution:
    def coinChange(self, coins, amount) -> int:
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