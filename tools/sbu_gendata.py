
import os
import sys
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

toolbar_width = 30


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def walk_sbu_data(data_root, valid_list):
    samples = []
    dataset = [[], []]
    for root, _, files in os.walk(data_root, topdown=False):
        samples.extend([os.path.join(root, f) for f in files
                        if os.path.splitext(f)[-1] == ".txt"])
    for s in samples:
        dataset[int(any([v in s for v in valid_list]))].append(s)

    return dataset


def gen_sbu_data(data_list, data_out_path, label_out_path, max_frame=46):
    sample_label = []

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(data_list), 3, max_frame, 15, 2))
    for i, data_name in enumerate(data_list):
        with open(data_name) as f:
            data = [list(map(float, l.split(',')[1:]))
                    for l in f.readlines() if bool(l.strip())]
        data = np.reshape(np.asarray(data), [-1, 2, 15, 3])
        # centralize
        data[..., 0:2] = data[..., 0:2] - 0.5
        data[..., -1] = data[..., -1] - 1
        data = np.transpose(data, (3, 0, 2, 1))
        label = int(data_name.split('/')[-3]) - 1
        print_toolbar(i * 1.0 / len(data_list),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(data_list)))
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((data_list, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path',
        default='/home/bob/Workspace/undergrad-project/dataset/SBU')
    parser.add_argument(
        '--out_folder',
        default='/home/bob/Workspace/undergrad-project/dataset/SBU')
    arg = parser.parse_args()
    part = ['train', 'val']
    for p, name in zip(part, walk_sbu_data(arg.data_path, ['s05'])):
        data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        gen_sbu_data(name, data_out_path, label_out_path)


