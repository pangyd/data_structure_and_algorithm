def numIslands(grid) -> int:
    """200.岛屿数量"""
    if not grid:
        return

    row = len(grid)
    col = len(grid[0])

    def dfs(i, j):
        grid[i][j] = "0"
        point = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for x, y in point:
            tmp_i = x + i
            tmp_j = y + j
            if 0 <= tmp_i < row and 0 <= tmp_j < col and grid[tmp_i][tmp_j] == "1":
                dfs(tmp_i, tmp_j)

    res = 0
    for i in range(row):
        for j in range(col):
            if grid[i][j] == "1":
                dfs(i, j)
                res += 1
    return res

def solve(board) -> None:
    """
    Do not return anything, modify board in-place instead.
    130.将包含边缘的区域变成B，再把内部区域变成X，最后把B变成O
    """
    def dfs(i, j):
        board[i][j] = "B"
        point = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for x, y in point:
            tmp_i = x + i
            tmp_j = y + j
            if 1 <= tmp_i < row and 1 <= tmp_j < col and board[tmp_i][tmp_j] == "O":
                dfs(tmp_i, tmp_j)

    row, col = len(board), len(board[0])

    for i in range(row):
        if board[i][0] == "O":
            dfs(i, 0)
        if board[i][col - 1] == "O":
            dfs(i, col - 1)

    for j in range(col):
        if board[0][j] == "O":
            dfs(0, j)
        if board[row - 1][j] == "O":
            dfs(row - 1, j)

    for i in range(row):
        for j in range(col):
            if board[i][j] == "O":
                board[i][j] = "X"
            if board[i][j] == "B":
                board[i][j] = "O"