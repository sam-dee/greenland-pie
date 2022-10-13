import numpy as np
from matplotlib import pyplot as plt

from datasets.mnist_dataset import Dataset


class DataLoader:
    def __init__(self, dataset: Dataset, num_classes, dataset_type, shuffle: bool, batch_size,
                 sample_type, epoch_size=None, probabilities=None):
        """
        :param dataset (Dataset): объект класса Dataset.
        :param nrof_classes (int): количество классов в датасете.
        :param dataset_type (string): (['train', 'valid', 'test']).
        :param shuffle (bool): нужно ли перемешивать данные после очередной эпохи.
        :param batch_size (int): размер батча.
        :param sample_type (string): (['default' - берем последовательно все данные, 'balanced' - сбалансированно,
        'prob' - сэмплирем с учетом указанных вероятностей])
        :param epoch_size (int or None): размер эпохи. Если None, необходимо посчитать размер эпохи (=размеру обучающей выюорки/batch_size)
        :param probabilities (array or None): в случае sample_type='prob' вероятности, с которыми будут выбраны элементы из каждого класса.
        """
        self.dataset = dataset
        if not batch_size:
            raise ValueError('Batch size must be > 0!')
        self.batch_size = batch_size

        if not num_classes:
            raise ValueError('Number of classes must be > 0!')
        self.num_classes = num_classes

        if sample_type not in ('default', 'balanced', 'prob'):
            raise ValueError(f"Sample type must be in ('default', 'balanced', 'prob'), got {sample_type}")
        self.sample_type = sample_type

        if dataset_type not in ['train', 'valid', 'test']:
            raise ValueError(f"Dataset type must be in ('train', 'valid', 'test'), got {dataset_type}")
        self.dataset_type = dataset_type

        if epoch_size is None:
            self.epoch_size = len(dataset) / batch_size

        self.shuffle = shuffle
        if self.sample_type == 'prob' and not probabilities:
            raise ValueError('Pass probabilities when using dataset_type = "prob"')

        if probabilities:
            if sum(probabilities) != 1:
                raise ValueError('Sum of probabilities must be 1')
            if len(probabilities) != num_classes:
                raise ValueError('Number of probabilities must eq to num_classes')
        self.probabilities = probabilities

        self.current_indexes = None

    def _default(self):
        indexes = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indexes)

        for i in range(0, len(indexes), self.batch_size):
            self.current_indexes = indexes[i: i + self.batch_size]

            yield self.dataset[self.current_indexes]

    def _balanced(self):
        raise NotImplemented

    def _prob(self):
        raise NotImplemented

    def _get_generator(self, gtype):
        match gtype:
            case 'default':
                return self._default
            case 'balanced':
                return self._balanced
            case 'prob':
                return self._prob
            case _:
                raise ValueError(f"Dataset type must be in ('default', 'balanced', 'prob'), got {gtype}")

    def batch_generator(self):
        """
        Создание батчей на эпоху с учетом указанного размера эпохи и типа сэмплирования.
        """
        generator = self._get_generator(self.sample_type)
        yield from generator()

    def show_batch(self, figsize=(7, 7)):
        """
        Необходимо визуализировать и сохранить изображения в батче (один батч - одно окно). Предварительно привести значение в промежуток
        [0, 255) и типу к uint8
        :return:
        """

        fig = plt.figure(figsize=figsize)

        for in_fig_img_index, idx in enumerate(self.current_indexes):
            transforms = list(self.dataset.step_transform(idx))
            num_imgs = len(self.current_indexes)
            num_augs = len(transforms)

            for t_idx, img in enumerate(transforms):
                fig.add_subplot(num_imgs, num_augs, 1 + in_fig_img_index * num_augs + t_idx)
                plt.imshow(img[0])
                plt.axis('off')
                plt.title(img[1])

        plt.show()
