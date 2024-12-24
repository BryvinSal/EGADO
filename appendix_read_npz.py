"""
Programmer: SAL
Date: 2023.12.06
"""
import numpy as np


filename = "D:\\User data\\sal\\NEW EGATDO MDM DEMUX\\EGA2D\\A2D_states.npz"
# Load data from the npz file
data = np.load(filename)

# Get all the keys (items) stored in the npz file
keys = data.files

# Access and print the contents of each item (array) stored in the npz file
for key in keys:
    # Access the arrays using their keys and print their contents
    result = data[key]
    print(f"Contents of '{key}':\n", data[key])

# Specific data
array = data['A2D_pixel_array']
ones = []
zeros = []
for i in range(len(array)):
    if array[i] > 0.99 and array[i] != 1.0:
        array[i] = 1
        ones.append(i)
    elif array[i] < 0.01 and array[i] != 0.0:
        array[i] = 0
        zeros.append(i)
    # else:
    #     print(array[i])
    #     assert 0

print(ones)
print(zeros)
np.savez("full_pixel_array.npz", pixel_array=array)

# Close the loaded data
data.close()
