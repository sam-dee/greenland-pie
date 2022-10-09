from utils.registry import Registry


def test_instantiate():
    name = 'Module register'
    r = Registry(name)
    assert r.name == name


def test_register():
    name = 'Module register'
    r = Registry(name)

    @r.register
    class Tst:
        ...

    @r.register
    class Tst1:
        ...

    try:
        @r.register
        class Tst:
            ...
    except Exception as e:
        assert type(e) == ValueError


def test_get():
    name = 'Module register'
    r = Registry(name)

    @r.register
    class Tst:
        ...

    tst_type = r.get('Tst', {})

    assert type(tst_type) == Tst
