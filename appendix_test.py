"""
Programmer: SAL
Date: 2023.12.05
"""
import numpy as np

def generate_B(K, A):
    rows, cols = K.shape
    B = []


    for col in range(cols):
        for row in range(rows):
            if K[row][col] in A:
                B.append(K[row][col])

    return B

# 示例矩阵 K 和向量 A
K = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

A = [3, 5, 7]

# 生成向量 B
B = generate_B(K, A)
print("生成的向量 B:", B)


# 定义矩阵的大小 MXN
M = 3
N = 4

# 生成一个向量，元素为0, 1, 2, 3... M*N-1
vector = np.arange(M * N)

# 将向量变形为MXN的矩阵
matrix = vector.reshape((M, N))

print("生成的向量：")
print(vector)

print("\n变形后的矩阵：")
print(matrix)

def find_values(matrix, value):
    positions = []
    for row_idx, row in enumerate(matrix):
        for col_idx, col in enumerate(row):
            if col == value:
                positions.append(row_idx * len(row) + col_idx)
    return positions

# 示例矩阵
matrix = np.array([
    [0, 0.5, 1, 0],
    [0, 1, 0.5, 0],
    [0.5, 0, 0, 1]
])

value_to_find = 1

# 查找值为0.5的元素位置信息
positions = find_values(matrix, value_to_find)
print("值为 1 的元素位置信息:", positions)