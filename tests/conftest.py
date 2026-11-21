import joblib
import pytest

from train import train


@pytest.fixture(scope="session")
def bundle():
    """A small model trained once for the whole test session."""
    bundle, _ = train(n_estimators=20)
    return bundle


@pytest.fixture
def client(bundle, tmp_path, monkeypatch):
    import app as app_module

    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setattr(app_module, "MODEL_PATH", str(path))
    monkeypatch.setattr(app_module, "_bundle", None)
    return app_module.app.test_client()
