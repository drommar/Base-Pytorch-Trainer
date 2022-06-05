import os
import shutil
import math
import numpy as np


class Splitter(object):
    def __init__(self, args):
        self.__args = args

    @staticmethod
    def __read_csv_labels(file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()[1:]
        tokens = [line.rstrip().split(',') for line in lines]
        return dict(((image_id, label) for image_id, label, variety, age in tokens))

    @staticmethod
    def __copyfile(file_name, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(file_name, target_dir)

    def __split_train_valid(self, valid_ratio):
        for label_dir in os.listdir(os.path.join(self.__args.dataset_dir, 'train_images')):
            index_of_this_label = 0
            total_num_of_this_label = len(os.listdir(os.path.join(self.__args.dataset_dir, 'train_images', label_dir)))
            for train_file in os.listdir(os.path.join(self.__args.dataset_dir, 'train_images', label_dir)):
                file_name = os.path.join(self.__args.dataset_dir, 'train_images', label_dir, train_file)
                label = label_dir
                self.__copyfile(file_name, os.path.join(self.__args.dataset_dir,
                                                        'train_valid_test', 'train_valid', label))
                valid_num_of_this_label = max(1, math.floor(total_num_of_this_label * valid_ratio))
                np.random.seed(self.__args.seed)
                shuffled_indices = np.random.permutation(total_num_of_this_label)
                test_indices = shuffled_indices[:valid_num_of_this_label]
                # train_indices = shuffled_indices[valid_num_of_this_label:]

                if index_of_this_label in test_indices:
                    self.__copyfile(file_name, os.path.join(self.__args.dataset_dir,
                                                            'train_valid_test', 'valid', label))
                else:
                    self.__copyfile(file_name, os.path.join(self.__args.dataset_dir,
                                                            'train_valid_test', 'train', label))
                index_of_this_label = index_of_this_label + 1

    def __organize_test(self):
        for test_file in os.listdir(os.path.join(self.__args.dataset_dir, 'test_images')):
            self.__copyfile(os.path.join(self.__args.dataset_dir, 'test_images', test_file),
                            os.path.join(self.__args.dataset_dir,
                                         'train_valid_test', 'test', 'unknown'))

    def split_dataset(self):
        self.__split_train_valid(self.__args.valid_ratio)
        self.__organize_test()


def main():
    pass


if __name__ == '__main__':
    main()
