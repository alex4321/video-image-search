import numpy as np
import imagehash
from PIL import Image


HISTOGRAM_BINS = 256
HISTORGRAM_MAX_DISSIMILARITY = 0.02
PHASH_HASH_SIZE = 20
PHASH_MAX_DISSIMILARITY = 0.06


def hist(frame):
    return np.histogram(frame.ravel(), HISTOGRAM_BINS)[0]


def hist_dissimilarity(hist1, hist2):
    shorter = min(np.linalg.norm(hist1), np.linalg.norm(hist2))
    diff_norm = np.linalg.norm(hist1 - hist2)
    return diff_norm / shorter


def phash(frame):
    return imagehash.phash(Image.fromarray(frame), hash_size=PHASH_HASH_SIZE).hash.ravel()


def phash_dissimilarity(hash1, hash2):
    if isinstance(hash1, list) or isinstance(hash2, list):
        pass
    assert hash1.shape == hash2.shape
    size = len(hash1)
    not_equal_count = np.logical_not(np.equal(hash1, hash2)).sum()
    return not_equal_count / size