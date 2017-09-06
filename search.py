import numpy as np
from funcs import phash, phash_dissimilarity, hist, hist_dissimilarity


CHECK_NEAREST_PHASHES = 300


def search(image, phash_indexes, hist_indexes):
    image_phash = phash(image)
    image_hist = hist(image)
    fname_hist_indexes = {}
    for index in hist_indexes:
        if index.fname not in fname_hist_indexes:
            fname_hist_indexes[index.fname] = []
        fname_hist_indexes[index.fname].append(index)

    phash_distances = np.array([
        phash_dissimilarity(image_phash, index.value)
        for i, index in enumerate(phash_indexes)
    ])
    similar_phash_indexes = np.argsort(phash_distances)[:CHECK_NEAREST_PHASHES]

    result = []
    for i in similar_phash_indexes:
        fname, frame = phash_indexes[i].fname, phash_indexes[i].frame
        hist_indexes = fname_hist_indexes[fname]
        frame_distances = np.abs(np.array([index.frame for index in hist_indexes]) - frame)
        nearest_frame_indexes = frame_distances.argsort()[:2]
        hist_indices_to_sort = [hist_indexes[nearest_frame_indexes[0]],
                                hist_indexes[nearest_frame_indexes[1]]]
        hist_distances = np.array([
            hist_dissimilarity(image_hist, index.value)
            for index in hist_indices_to_sort
        ])
        hist_indexes = hist_distances.argsort()
        for index in hist_indexes:
            item = (hist_indices_to_sort[index].fname, hist_indices_to_sort[index].frame)
            if item not in result:
                result.append(item)
    return result