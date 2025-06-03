import os
import random

def get_file_names(folder_path, prefix):
    file_names = []
    if os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                file_names.append(prefix + '/' + item)
    else:
        print(f"The folder path {folder_path} does not exist or is not a directory.")
    return file_names

def split_list_randomly(input_list):
    random.shuffle(input_list)
    p1_end = int(len(input_list) * 0.5)
    p2_end = int(len(input_list) * 0.6)
    part1 = input_list[:p1_end]
    part2 = input_list[p1_end:p2_end]
    part3 = input_list[p2_end:]
    return part1, part2, part3

def write_list_to_file(file_names, output_file):
    with open(output_file, 'w') as file:
        for name in file_names:
            file.write(name + '\n')

def create_splits(subfolders, file_names, save_path='./CCPD2019/new_splits/'):
    os.makedirs(save_path, exist_ok=True)
    train = []
    validate = []
    for i in range(len(subfolders)):
        p1, p2, p3 = split_list_randomly(file_names[i])
        train.extend(p1)
        validate.extend(p2)
        write_list_to_file(p3, save_path + 'test_' + subfolders[i] + '.txt')
    write_list_to_file(train, save_path + 'train.txt')
    write_list_to_file(validate, save_path + 'validate.txt')

def check_splits(subfolders, save_path='./CCPD2019/new_splits/'):
    train_file = os.path.join(save_path, 'train.txt')
    validate_file = os.path.join(save_path, 'validate.txt')
    test_files = [os.path.join(save_path, f'test_{subfolder}.txt') for subfolder in subfolders]

    if not os.path.exists(train_file) or not os.path.exists(validate_file) or not all(os.path.exists(f) for f in test_files):
        return False
    return True

def generate_splits():
    """
    Generates train, validate, and test splits for the CCPD2019 dataset.
    If the splits already exist, it does nothing.
    """
    random.seed(42)
    subfolders = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather']
    if check_splits(subfolders, save_path='./CCPD2019/new_splits/'):
        return
    file_names = []
    for subfolder in subfolders:
        result = get_file_names('./CCPD2019/'+subfolder, subfolder)
        file_names.append(result)

    create_splits(subfolders, file_names, save_path='./CCPD2019/new_splits/')