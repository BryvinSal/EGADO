import numpy as np
from skimage import feature
import os
import matplotlib.pyplot as plt


def convol(matrix, kernel_size, stride, padding):
    m, n = matrix.shape
    kernel = np.ones((kernel_size, kernel_size))

    padded_matrix = np.pad(matrix, ((padding, padding), (padding, padding)), mode='constant')

    output_size = ((padded_matrix.shape - np.array(kernel.shape)) // stride) + 1
    result = np.zeros(output_size, dtype=int)

    for i in range(0, m + 2 * padding - kernel_size + 1, stride):
        for j in range(0, n + 2 * padding - kernel_size + 1, stride):
            window = padded_matrix[i:i + kernel_size, j:j + kernel_size]
            result[i // stride, j // stride] = int(np.sum(window * kernel) > 0)

    conversion_rate = 100 - (100 * np.sum(result) / result.size)
    print(f"Direct Conversion Rate = {conversion_rate:.2f}%")
    return result


def pattern_mapping(A, B, value, scale):
    new_rows = A.shape[0] // scale
    new_cols = A.shape[1] // scale
    new_matrix = np.zeros((new_rows, new_cols))

    for i in range(new_rows):
        for j in range(new_cols):
            row_range = slice(i * scale, (i + 1) * scale)
            col_range = slice(j * scale, (j + 1) * scale)

            center_row = (row_range.start + row_range.stop - 1) // 2
            center_col = (col_range.start + col_range.stop - 1) // 2

            new_matrix[i, j] = A[center_row, center_col]

    result_matrix = np.where(B == 0, new_matrix, value)
    return result_matrix


def get_conversion_matrix(analog_file, target_folder, pixel_size, mesh_size=20, buff=0):
    """
    :param analog_file: final npz file of topology optimization
    :param target_folder: folder of eg-atd files
    :param pixel_size: size of atd pixel
    :param mesh_size: size of to mesh grid
    :param buff: value = 0,2,4... larger buff means smaller conversion rate
    :return: edge decision matrix for eg-atd
    """
    print("Now we are generating the conversion matrix ...")
    data = np.load(analog_file)

    x = data['x']
    y = data['y']
    z = data['z']
    eps = data['eps']

    full_eps = np.broadcast_to(eps[:, :, None], (len(x), len(y), len(z)))

    analog_pattern = full_eps[:, :, 6]
    analog_pattern = analog_pattern.T[:-1, :-1]
    analog_pattern = -1 * np.flip(analog_pattern, axis=0)
    max_eps = np.max(analog_pattern)
    min_eps = np.min(analog_pattern)
    analog_pattern = (analog_pattern - min_eps) / (max_eps - min_eps)

    edge_img = feature.canny(analog_pattern)

    pixel_size = np.ceil(pixel_size*1e9)
    # print(pixel_size)
    scale = int(pixel_size / mesh_size)
    decision_area = scale + buff
    # print(f"Scale={scale}")

    conv_result = convol(edge_img, decision_area, scale, buff // 2)
    mapped_result = pattern_mapping(analog_pattern, conv_result, 0.5, scale)

    # plot the original analog pattern and converted pattern
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

    axes[0].imshow(analog_pattern, cmap='bone', vmin=0, vmax=1)
    axes[0].axis('off')
    axes[0].set_title('Original Pattern')

    axes[1].imshow(mapped_result, cmap='bone', vmin=0, vmax=1)
    axes[1].axis('off')
    axes[1].set_title('Conversion Result')
    plt.tight_layout()

    plt.savefig(os.path.join(target_folder, "pattern conversion.png"))
    plt.close()

    np.savez(os.path.join(target_folder, "conversion_matrix.npz"), conversion_matrix=mapped_result, scale=scale,
             buff=buff)

    return mapped_result
