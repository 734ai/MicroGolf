def solve(grid):
    """Test function for optimization"""
    result =[]
    for i in range(len(grid)):
        row =[]
        for j in range(len(grid[0])):
            if grid[i][j]==0:
                value = 1
            else:
                value = 0
            row.append(value)
        result.append(row)
    return result

def another_function(x,y):
    temp_var = x
    if temp_var>y:
        return 1
    else:
        return 0
