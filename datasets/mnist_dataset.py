import gzip
import json
from enum import Enum
from typing import List, Callable, Iterable

import numpy as np


class DatasetType(Enum):
    train = 'train'
    valid = 'valid'
    test = 'test'


class Dataset:
    def __init__(
            self, data_path: str, dataset_type: DatasetType, transforms: List[Callable], num_classes: int,
            *, ifname='train-images-idx3-ubyte.gz', lfname='train-labels-idx1-ubyte.gz'
    ):
        """
        :param data_path: path to data dir. Files assumed to be in .gz
        :param dataset_type: (['train', 'valid', 'test']).
        :param transforms: image transformations
        :param num_classes: number of classes
        :param ifname: use conventional img filename
        :param lfname: use conventional labels filename

        see https://pypi.org/project/python-mnist/
        """
        self._images = []
        self._labels = []

        self._data_path = data_path
        self._ifname = ifname
        self._lfname = lfname

        self._dataset_type = dataset_type
        self._transforms = transforms
        self._num_classes = num_classes

        self._stats = None

    def _read_labels(self):
        with gzip.open(self._data_path + self._lfname, 'r') as f:
            _ = int.from_bytes(f.read(4), 'big')  # magic number
            _ = int.from_bytes(f.read(4), 'big')  # number of labels
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)

            return labels

    def _read_images(self):
        with gzip.open(self._data_path + self._ifname, 'r') as f:
            _ = int.from_bytes(f.read(4), 'big')  # magic number
            image_count = int.from_bytes(f.read(4), 'big')  # number of images
            row_count = int.from_bytes(f.read(4), 'big')  # row count
            column_count = int.from_bytes(f.read(4), 'big')  # column count
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8) \
                .reshape((image_count, row_count, column_count))

            return images

    def read_data(self, no_stat=False):
        self._labels = self._read_labels()
        self._images = self._read_images()

        if not no_stat:
            self.show_statistics()

    def __len__(self):
        """
        :return: sample data length
        """
        return len(self._images)

    def one_hot_labels(self, label):
        """
        for 10 classes label 5-> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        :param label: class label
        :return: one-hot encoding vector
        """
        if label > self._num_classes:
            raise ValueError(f'Label {label} not found in classes!')
        return [1 if el == label else 0 for el in range(self._num_classes)]

    def __getitem__(self, idx):
        """
        :param idx: element index in sample data
        :return: preprocessed image and label
        """
        images = self._images[idx]
        labels = self._labels[idx]
        if isinstance(idx, Iterable):
            for image in images:
                for transform in self._transforms:
                    images = transform(image)
        else:
            for transform in self._transforms:
                images = transform(images)

        return images, labels

    def step_transform(self, idx):
        image = self._images[idx]
        label = self._labels[idx]
        yield image, label
        if self._transforms:
            for transform in self._transforms:
                image = transform(image)
                yield image, label

    def show_statistics(self):
        """
        Необходимо вывести количество элементов в датасете, количество классов и количество элементов в каждом классе
        """
        count = self.__len__()
        by_class = {c: 0 for c in range(self._num_classes)}
        for label in self._labels:
            if label in by_class:
                by_class[label] += 1
            else:
                print(f'Element {label} not found in classes!')

        print(f'[sample data length is {count} | number of classes is {self._num_classes}]')
        print(tuple(by_class.items()))
