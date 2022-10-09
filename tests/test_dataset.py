import pytest

from datasets.mnist_dataset import Dataset
from utils.config import Config


@pytest.fixture(scope='session')
def ds():
    cfg = Config.from_file('../configs/tests/test_dataset.json')
    ds = Dataset(**vars(cfg.data))

    return ds


def test_read_data(ds: Dataset):
    ds.read_data()


def test_one_hot_labels(ds: Dataset):
    assert ds.one_hot_labels(0) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert ds.one_hot_labels(9) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    try:
        assert ds.one_hot_labels(13) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    except Exception as e:
        assert type(e) == ValueError


def test_get_item(ds: Dataset):
    ds.read_data()
    img, label = ds[0]

    assert label is not None
