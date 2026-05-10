import numpy as np

from deep_har.preprocess import ChannelStandardizer


def test_channel_standardizer_roundtrip(tmp_path):
    X = np.random.default_rng(0).normal(size=(20, 128, 9)).astype("float32")
    scaler = ChannelStandardizer().fit(X)
    Xt = scaler.transform(X)
    assert Xt.shape == X.shape
    assert np.allclose(Xt.mean(axis=(0, 1)), 0, atol=1e-5)
    assert np.allclose(Xt.std(axis=(0, 1)), 1, atol=1e-5)

    path = tmp_path / "normalizer.npz"
    scaler.save(path)
    loaded = ChannelStandardizer.load(path)
    assert np.allclose(loaded.transform(X), Xt)
