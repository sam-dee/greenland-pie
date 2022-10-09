import numpy as np
from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self, dataset, num_classes, dataset_type, shuffle, batch_size,
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
        self.batch_size = batch_size

        self.current_indexes = None

    def batch_generator(self):
        """
        Создание батчей на эпоху с учетом указанного размера эпохи и типа сэмплирования.
        """
        indexes = np.arange(len(self.dataset))
        np.random.shuffle(indexes)
        for i in range(0, len(indexes), self.batch_size):
            self.current_indexes = indexes[i: i + self.batch_size]

            yield self.dataset[self.current_indexes]

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
