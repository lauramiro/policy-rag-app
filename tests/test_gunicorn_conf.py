import importlib.util
import pathlib

CONF_PATH = pathlib.Path(__file__).parent.parent / "gunicorn.conf.py"


def _load_conf():
    spec = importlib.util.spec_from_file_location("gunicorn_conf", CONF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conf_file_exists():
    assert CONF_PATH.exists()


def test_timeout_covers_model_load():
    conf = _load_conf()
    assert conf.timeout >= 120


def test_warmup_hook_defined():
    conf = _load_conf()
    assert callable(conf.post_worker_init)


def test_warmup_failure_does_not_raise(monkeypatch):
    conf = _load_conf()

    class FakeLog:
        def __init__(self):
            self.exception_called = False

        def info(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            self.exception_called = True

    class FakeWorker:
        log = FakeLog()

    import app as app_module

    def _boom():
        raise RuntimeError("warmup blew up")

    monkeypatch.setattr(app_module, "get_resources", _boom)

    worker = FakeWorker()
    conf.post_worker_init(worker)

    assert worker.log.exception_called
