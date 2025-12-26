import numpy as np
from PIL import Image

from star_wars_characters.utils.images import preprocess_pil, softmax


def test_preprocess_shape():
    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))
    x = preprocess_pil(img, image_size=224)
    assert x.shape == (3, 224, 224)
    assert x.dtype == np.float32


def test_softmax():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    p = softmax(x, axis=1)
    assert p.shape == (1, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
