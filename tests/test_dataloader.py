import pytest

from dataloader.dataloader import DataLoader
from datasets.mnist_dataset import Dataset
from ops.transformations import transform_registry
from utils.config import Config


@pytest.fixture(scope='session')
def ds():
    cfg = Config.from_file('../configs/tests/test_dataset.json')
    params = vars(cfg.data)
    params['transforms'] = [transform_registry.get('GaussianNoise', {'sigma': 255, 'mean': 0})]
    ds = Dataset(**params)

    return ds


def test_transforms():
    cfg = Config.from_file('../configs/tests/test_dataloader.json')
    params = vars(cfg.data)

    transforms = []
    for transform in params['transforms']:
        for name, kwargs in transform.items():
            transforms.append(transform_registry.get(name, vars(kwargs)))
    params['transforms'] = transforms

    ds = Dataset(**params)

    dl = DataLoader(ds, 10, 'test', True, 4, 'default')
    ds.read_data()

    next(dl.batch_generator())
    dl.show_batch()
