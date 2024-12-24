import os
import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import gaussian_filter, zoom, rotate
from scipy.ndimage import gaussian_filter


def npztomatlab(target_file, name, array):
    from scipy.io import savemat
    variables = {name: array.astype(np.double)}
    savemat(target_file, variables)


def generate_blurred_noise(design_area_size_x, design_area_size_y, bias=0.5, range_=0.5, target_folder=os.getcwd(),
                           name=""):
    # bias, range control the average value and the range of the initial noise,
    # but the final range is unknown and the final average may fluctuate around bias.
    # Generally, the final average is close to bias and the final range is limited in range_.
    # 这里不需要缩放是因为，高斯滤波等操作不会改变平均值和范围，只会改变分布，我们只需要设定好最开始的uniform_noise的范围即可。
    os.makedirs(target_folder, exist_ok=True)
    uniform_noise = np.random.uniform(bias - range_ / 2, bias + range_ / 2, (design_area_size_x, design_area_size_y))
    sigma = np.random.uniform(1.0, 2.0)
    blurred_noise = gaussian_filter(uniform_noise, sigma)  # 高斯滤波保持平均值基本不变
    # print(sigma, np.min(blurred_noise), np.max(blurred_noise))

    # Apply random zoom and rotation，基本保证平均值不变
    zoom_factor = np.random.uniform(1.1, 2.0)
    rot_angle = np.random.uniform(0, 360)
    zoomed_array = zoom(blurred_noise, zoom=zoom_factor)
    rotated_array = rotate(zoomed_array, angle=rot_angle, reshape=True,
                           mode='reflect')  # mode='reflect' to avoid black borders after rotation
    # Crop the array to the original size
    center_x = rotated_array.shape[0] // 2
    center_y = rotated_array.shape[1] // 2
    half_size_x = design_area_size_x // 2
    half_size_y = design_area_size_y // 2

    if design_area_size_x % 2 == 0:
        start_x = center_x - half_size_x
        end_x = center_x + half_size_x - 1
    else:
        start_x = center_x - half_size_x
        end_x = center_x + half_size_x

    if design_area_size_y % 2 == 0:
        start_y = center_y - half_size_y
        end_y = center_y + half_size_y - 1
    else:
        start_y = center_y - half_size_y
        end_y = center_y + half_size_y

    centered_array = rotated_array[start_x:end_x + 1, start_y:end_y + 1]

    # Resize the array back to the original size using interpolation
    final_noise = np.clip(centered_array, 0, 1)
    average = np.mean(final_noise)
    min = np.min(final_noise)
    max = np.max(final_noise)
    # final_noise = bias + (final_noise - np.mean(final_noise)) * (1 - bias)
    plt.figure()
    plt.imshow(final_noise, cmap='Greys', interpolation='none', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Blurred Noise")
    plt.savefig(
        target_folder + '/Blurred_Noise_{}_a{:.3f}_min{:.3f}_max{:.3f}_sigma_{:3f}.png'.format(name, average, min, max,
                                                                                               sigma))
    np.savez(
        target_folder + '/Blurred_Noise_{}_a{:.3f}_min{:.3f}_max{:.3f}_sigma_{:3f}.npz'.format(name, average, min, max,
                                                                                               sigma),
        bias=bias, range=range_, sigma=sigma, zoom_factor=zoom_factor, rot_angle=rot_angle, average=average, min=min,
        max=max,
        blurred_noise=final_noise)
    return final_noise


# for i in range(50):
#     generate_blurred_noise(100, 100, 1, 0.1, target_folder=os.path.join(os.getcwd(),"gaussian_noise_{}".format(0)), name=i)

def scale_and_offset_noise(noise, target_mean, target_range):
    current_mean = np.mean(noise)
    current_range = np.max(noise) - np.min(noise)

    scaled_noise = (noise - current_mean) * (target_range / current_range) + target_mean
    return np.clip(scaled_noise, 0, 1)


def generate_perlin_noise(design_area_size_x, design_area_size_y, bias=0.5, range_=0.5, target_folder=os.getcwd(),
                          name=""):
    os.makedirs(target_folder, exist_ok=True)
    perlin_noise = np.zeros((design_area_size_x, design_area_size_y))
    seed = np.random.randint(0, 500)  # 随机种子，超过500好像会出现横状条纹
    freq_factor = np.random.uniform(0.01, 0.09)  # 频率太高会有很多细节，不利于优化
    for i in range(design_area_size_x):
        for j in range(design_area_size_y):
            perlin_noise[i, j] = pnoise2(i * freq_factor, j * freq_factor, base=seed)
    # 这里需要缩放是因为，我们不知道这个最后会产生什么样的效果。所以我们需要把它缩放到一个合适的范围内。
    final_noise = scale_and_offset_noise(perlin_noise, bias, range_)
    average = np.mean(final_noise)
    min = np.min(final_noise)
    max = np.max(final_noise)
    plt.figure()
    plt.imshow(final_noise, cmap='Greys', interpolation='none', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Perlin Noise")
    plt.savefig(
        target_folder + '/Perlin_Noise_{}_a{:.3f}_min{:.3f}_max{:.3f}_f{:.3f}_s{:3d}.png'.format(name, average, min,
                                                                                                 max, freq_factor,
                                                                                                 seed))
    np.savez(target_folder + '/Perlin_Noise_{}_a{:.3f}_min{:.3f}_max{:.3f}_f{:.3f}_s{:3d}.npz'.
             format(name, average, min, max, freq_factor, seed), bias=bias, range=range_, seed=seed,
             freq_factor=freq_factor, average=average, min=min, max=max, perlin_noise=final_noise)
    return final_noise

