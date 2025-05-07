import os
import numpy as np
from ase.build import make_supercell
from ase.io import read, write
import joblib
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DeepCIF_model import convnext

name_list = ['PCN_66.cif']  # File name of CIF(supports batch input)
label = [304]  # Adsorption values label of CIF at 298K, 35bar

# Data preprocessing
name_explore = []
name_process = []
name_after = []
for item in name_list:
    name_explore.append(item.split(".")[0] + "_explore.cif")
    name_process.append("process_" + item)
    name_after.append(item.split(".")[0] + "_after.pickle")

for per_name in range(len(name_list)):
    atoms = read(name_list[per_name])
    supersize = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    explore_atoms = make_supercell(atoms, supersize)
    write(name_explore[per_name], explore_atoms, format="cif")
    # 计算process版本
    new_cif_name = name_process[per_name]
    new_cif = open(new_cif_name, 'w')
    transfrom_matrix = np.zeros([3, 3])
    sub_dir = name_explore[per_name]
    if os.path.isfile(sub_dir):
        file = open(sub_dir)
    for ii in range(0, 3):
        line = file.readline()
    line = file.readline()
    line_vec = list(filter(None, line.strip("\n").split(" ")))
    length_a = float(line_vec[1])
    line = file.readline()
    line_vec = list(filter(None, line.strip("\n").split(" ")))
    length_b = float(line_vec[1])
    line = file.readline()
    line_vec = list(filter(None, line.strip("\n").split(" ")))
    length_c = float(line_vec[1])
    line = file.readline()
    line_vec = list(filter(None, line.strip("\n").split(" ")))
    angle_alpha = float(line_vec[1])
    line = file.readline()
    line_vec = list(filter(None, line.strip("\n").split(" ")))
    angle_beta = float(line_vec[1])
    line = file.readline()
    line_vec = list(filter(None, line.strip("\n").split(" ")))
    angle_gamma = float(line_vec[1])
    new_line = str(length_a) + '\t' + str(length_b) + '\t' + str(length_c) + '\t' + str(
        angle_alpha) + '\t' + str(angle_beta) + '\t' + str(angle_gamma)
    new_cif.write(new_line + '\n')
    for ii in range(10, 26):
        line = file.readline()
    line = file.readline()
    while line:
        line_vec = list(filter(None, line.strip("\n").split(" ")))
        abc_point = np.array([float(line_vec[3]), float(line_vec[4]), float(line_vec[5])])
        while abc_point[0] > 1 or abc_point[0] < 0:
            if abc_point[0] > 1:
                abc_point[0] -= 1
            else:
                abc_point[0] += 1
        while abc_point[1] > 1 or abc_point[1] < 0:
            if abc_point[1] > 1:
                abc_point[1] -= 1
            else:
                abc_point[1] += 1
        while abc_point[2] > 1 or abc_point[2] < 0:
            if abc_point[2] > 1:
                abc_point[2] -= 1
            else:
                abc_point[2] += 1
        new_line = str(line_vec[0]) + '\t' + str(
            round(float(abc_point[0]), 2)) + '\t' + str(
            round(float(abc_point[1]), 2)) + '\t' + str(
            round(float(abc_point[2]), 2))
        new_cif.write(new_line + '\n')
        line = file.readline()
    new_cif.close()
    file.close()

    # 映射装载
    sub_dir = name_process[per_name]
    atoms = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17, "V": 23, "Cu": 29, "Zn": 30, "Br": 35, "Zr": 40}
    atoms_index = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "Cl": 5, "V": 6, "Cu": 7, "Zn": 8, "Br": 9, "Zr": 10}
    results_x = []
    result_x = 0
    a = np.zeros((11, 51, 51), dtype=np.uint8)
    b = np.zeros((11, 51, 51), dtype=np.uint8)
    c = np.zeros((11, 51, 51), dtype=np.uint8)

    if os.path.isfile(sub_dir):
        file = open(sub_dir)
    line = file.readline()
    line_vec = line.strip("\n").split("\t")
    length_a = float(line_vec[0])
    length_b = float(line_vec[1])
    length_c = float(line_vec[2])
    angle_a = float(line_vec[3])
    angle_b = float(line_vec[4])
    angle_c = float(line_vec[5])

    line = file.readline()
    while line:
        line_vec = line.strip("\n").split("\t")
        line_vec[1] = round(float(line_vec[1]) * 50)
        line_vec[2] = round(float(line_vec[2]) * 50)
        line_vec[3] = round(float(line_vec[3]) * 50)
        per_atoms_index = int(atoms_index[line_vec[0]])
        a[per_atoms_index][line_vec[1]][line_vec[2]] = int(
            atoms[line_vec[0]] + a[per_atoms_index][line_vec[1]][line_vec[2]])
        b[per_atoms_index][line_vec[2]][line_vec[3]] = int(
            atoms[line_vec[0]] + b[per_atoms_index][line_vec[2]][line_vec[3]])
        c[per_atoms_index][line_vec[3]][line_vec[1]] = int(
            atoms[line_vec[0]] + c[per_atoms_index][line_vec[3]][line_vec[1]])
        line = file.readline()
    file.close()
    x_batch = np.empty((1, 3, 11, 51, 51), dtype=np.uint8)
    cell_numpy = np.empty((6,))
    x_batch[0, 0:, :, :, :] = a.reshape(1, 1, 11, 51, 51)
    x_batch[0, 1:, :, :, :] = b.reshape(1, 1, 11, 51, 51)
    x_batch[0, 2:, :, :, :] = c.reshape(1, 1, 11, 51, 51)

    cell_numpy[0] = length_a
    cell_numpy[1] = length_b
    cell_numpy[2] = length_c
    cell_numpy[3] = angle_a
    cell_numpy[4] = angle_b
    cell_numpy[5] = angle_c
    result_x = [x_batch, cell_numpy]
    results_x.append(result_x)
    joblib.dump(results_x, name_after[per_name])

    # Start prediction
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    test_dataset = joblib.load(name_after[per_name])
    x_test = test_dataset
    test_data_size = len(x_test)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False, drop_last=True)
    net = convnext()
    net.to(device)
    save_path = 'DeepCIF_CH4_298_35'
    net.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    net.eval()
    x1 = []
    y1 = []
    error = 0

    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout)
    for test_data in val_bar:
        x_batch_test = test_data
        xyz_batch_process = x_batch_test[0]
        cell_batch_process = x_batch_test[1]
        xyz_batch = torch.reshape(xyz_batch_process, (1, 3, 11, 51, 51)).float().to(device)
        cell_batch = torch.reshape(cell_batch_process, (1, 6)).float().to(device)
        outputs = net(xyz_batch, cell_batch)
        true = label[per_name]
        predict = float(outputs) * 524.8945546
        print(name_list[per_name], "predict:", predict)
        temp = (true - predict)
        print(name_list[per_name], "error:", temp)
