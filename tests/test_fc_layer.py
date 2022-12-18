import pytest

from dataloader.dataloader import DataLoader
from datasets.mnist_dataset import Dataset
from layers.fc_layer import FCLayer
from utils.config import Config


@pytest.fixture(scope='session')
def ds():
    cfg = Config.from_file('../configs/tests/test_dataset.json')
    ds = Dataset(**vars(cfg.data))

    return ds


def test_read_data(ds: Dataset):
    N = 3
    dl = DataLoader(ds, 10, 'train', True, 9, 'default')
    ds.read_data()

    l1 = FCLayer(28*28, 10)

    for _ in range(N):
        batch = next(dl.batch_generator())
        print(batch)
