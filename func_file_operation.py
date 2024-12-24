import shutil
from func_generate_noise import *


def copy_specified_filename_to(target_folder, filename="_A2D."):
    # 获取当前文件夹路径
    current_folder = os.getcwd()
    os.makedirs(target_folder, exist_ok=True)
    move_success = False

    # 遍历当前文件夹下的所有文件
    while 1:
        for file in os.listdir(current_folder):
            # 检查文件名是否包含 "_A2D" 字符
            if filename in file:
                # 构建源文件和目标文件的路径
                source_path = os.path.join(current_folder, file)
                target_path = os.path.join(target_folder, file)
                shutil.copy(source_path, target_path)
        # 检查
        for file in os.listdir(target_folder):
            # 检查文件名是否包含 "_A2D" 字符
            if filename in file:
                # 构建源文件和目标文件的路径
                move_success = True
        if move_success:
            break


def move_files_to(original_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for file in os.listdir(original_folder):
        file_path = os.path.join(original_folder, file)
        # 判断是否为文件而不是文件夹，并且不是 data1 和 data 文件夹
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(target_folder, file))


def export_TO_param_for_a2d(working_dir, number=1, type="MAX"):
    param_path = working_dir
    # 获取文件夹中所有文件的列表
    file_list = os.listdir(param_path)
    # 过滤出以'a_'开头且以'.npy'结尾的文件
    filtered_files = [file for file in file_list if
                      file.startswith('parameters_') and file.endswith('.npz')]
    # 如果有匹配的文件
    if filtered_files:
        # 提取后缀数字并排序，选取后缀最大的文件
        if type == "MAX":
            suffix_file = max(filtered_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        elif type == "MIN":
            suffix_file = min(filtered_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        elif type == "specified":
            suffix_file = "parameters_{}.npz".format(number)
        else:
            assert False, "type should be MAX or MIN"
        return os.path.join(param_path, suffix_file)
    else:
        assert False, "No parameter file found in {}".format(param_path)


def is_list_with_two_numbers(variable):
    if isinstance(variable, list) and len(variable) == 2 and all(
            isinstance(item, (int, float)) for item in variable):
        return True
    return False
