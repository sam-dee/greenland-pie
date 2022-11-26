import random
from abc import ABC, abstractmethod

import numpy as np

from utils.registry import Registry

transform_registry = Registry('Transform Registry')


class Transformation(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


@transform_registry.register
class Pad(Transformation):
    def __init__(self, image_size=0, fill=0, mode='constant'):
        """
        :param image_size (int or tuple): размер итогового изображения. Если одно число, на выходе будет
        квадратное изображение. Если 2 числа - прямоугольное.
        :param fill (int or tuple): значение, которым будет заполнены поля. Если одно число, все каналы будут заполнены
        этим числом. Если 3 - соответственно по каналам.
        :param mode (string): тип заполнения:
        constant: все поля будут заполнены значение fill;
        edge: все поля будут заполнены пикселями на границе;
        reflect: отображение изображения по краям (прим. [1, 2, 3, 4] => [3, 2, 1, 2, 3, 4, 3, 2])
        symmetric: симметричное отображение изображения по краям (прим. [1, 2, 3, 4] => [2, 1, 1, 2, 3, 4, 4, 3])
        """
        if mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        self.image_size = image_size
        self.fill = fill
        self.mode = mode

    def __call__(self, img):
        return np.pad(img, (self.image_size, self.fill), mode=self.mode)


class Translate(Transformation):
    def __init__(self, shift=10, direction='right', roll=True):
        """
        :param shift (int): количество пикселей, на которое необходимо сдвинуть изображение
        :param direction (string): направление (['right', 'left', 'down', 'up'])
        :param roll (bool): Если False, не заполняем оставшуюся часть. Если True, заполняем оставшимся краем.
        (прим. False: [1, 2, 3]=>[0, 1, 2]; True: [1, 2, 3] => [3, 1, 2])
        """
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented


# TODO:
@transform_registry.register
class Scale(Transformation):
    def __init__(self, image_size, scale):
        """
        :param image_size (int): размер вырезанного изображения (по центру).
        :param scale (float): во сколько раз увеличить изображение.
        """
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented


# TODO:
class Crop(Transformation, ABC):
    pass


@transform_registry.register
class RandomCrop(Crop):
    def __init__(self, crop_size):
        """
        :param crop_size (int or tuple): размер вырезанного изображения.
        """
        self.crop_size = crop_size

    def __call__(self, img):
        if type(self.crop_size) is int:
            cropx = self.crop_size
            cropy = self.crop_size
        else:
            cropx, cropy = self.crop_size

        y, x = img.shape
        startx = max(random.randint(0, x) // 2 - (cropx // 2), 0)
        starty = max(random.randint(0, y) // 2 - (cropy // 2), 0)
        return img[starty:starty + cropy, startx:startx + cropx]


@transform_registry.register
class CenterCrop(Crop):
    def __init__(self, crop_size):
        """
        :param crop_size (int or tuple): размер вырезанного изображения (вырезать по центру).
        """
        self.crop_size = crop_size

    def __call__(self, img):
        if type(self.crop_size) is int:
            cropx = self.crop_size
            cropy = self.crop_size
        else:
            cropx, cropy = self.crop_size

        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]


@transform_registry.register
class RandomRotateImage(Transformation):
    def __init__(self, min_angle, max_angle):
        """
        :param min_angle (int): минимальный угол поворота.
        :param max_angle (int): максимальный угол поворота.
        Угол поворота должен быть выбран равномерно из заданного промежутка.
        """
        self.min_angle = min_angle
        self.max_angle = max_angle
        pass

    def __call__(self, img):
        a = self.change_angle_to_radius_unit(random.randint(self.min_angle, self.max_angle))

        rotation_mat = np.transpose(np.array([[np.cos(a), -np.sin(a)],
                                              [np.sin(a), np.cos(a)]]))
        h, w = img.shape

        pivot_point_x = w / 2
        pivot_point_y = h / 2

        new_img = np.zeros(img.shape, dtype='u1')

        for height in range(h):
            for width in range(w):
                xy_mat = np.array([[width - pivot_point_x], [height - pivot_point_y]])

                rotate_mat = np.dot(rotation_mat, xy_mat)

                new_x = int(pivot_point_x + rotate_mat[0])
                new_y = int(pivot_point_y + rotate_mat[1])

                if (0 <= new_x <= w - 1) and (0 <= new_y <= h - 1):
                    new_img[new_y, new_x] = img[height, width]

        return new_img

    def change_angle_to_radius_unit(self, angle):  # noqa
        angle_radius = angle * (np.pi / 180)
        return angle_radius


@transform_registry.register
class GaussianNoise(Transformation):
    def __init__(self, mean=0, sigma=0.03, by_channel=False):
        """
        :param mean (int): среднее значение.
        :param sigma (int): максимальное значение ско. Итоговое значение должно быть выбрано равномерно в промежутке
        [0, sigma].
        :param by_channel (bool): если True, то по каналам независимо.
        """
        self.mean = mean
        self.sigma = sigma

    def __call__(self, img):
        row, col = img.shape

        gauss = np.random.normal(self.mean, self.sigma, (row, col))
        gauss = gauss.reshape((row, col))
        noisy = img + gauss

        return noisy


class Salt(Transformation):
    def __init__(self, prob, by_channel=False):
        """
        :param prob (float): вероятность, с которой пиксели будут заполнены белым.
        :param by_channel (bool): если True, то по каналам независимо.
        """
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented


class Pepper(Transformation):
    def __init__(self, prob, by_channel=False):
        """
        :param prob (float): вероятность, с которой пиксели будут заполнены черным.
        :param by_channel (bool): если True, то по каналам независимо.
        """
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented


class ChangeBrightness(Transformation):
    def __init__(self, value=30, type='brightness'):
        """
        :param value (int): насколько изменить яркость. Аналогично hue, contrast, saturation.
        :param type (string): один из [brightness, hue, contrast, saturation].
        """
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented


class GaussianBlur(Transformation):
    def __init__(self, ksize=(5, 5)):
        """
        :param ksize (tuple): размер фильтра.
        """
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented


class Normalize(Transformation):
    def __init__(self, mean=128, var=255):
        """
        :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
        :param var (int): значение, на которое необходимо поделить.
        """
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented
