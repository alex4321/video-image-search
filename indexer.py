import numpy as np
from funcs import phash, phash_dissimilarity, hist, hist_dissimilarity, HISTORGRAM_MAX_DISSIMILARITY, PHASH_MAX_DISSIMILARITY
from index import FrameIndex


class VideoIndexer:
    def __init__(self, fname, dataset):
        self.fname = fname
        self.dataset = dataset

    def _hist_frame_index(self, histogram_index, frame, frame_index):
        histogram = hist(frame)
        if len(histogram_index) == 0:
            histogram_index.append((frame_index, histogram, [histogram]))
        else:
            previous_frame_index, previous_histogram, previous_histograms = histogram_index[-1]
            if hist_dissimilarity(histogram, previous_histogram) > HISTORGRAM_MAX_DISSIMILARITY:
                previous_histograms_mean = np.array(previous_histograms).mean(axis=0)
                del histogram_index[-1]
                histogram_index.append((previous_frame_index, previous_histogram, previous_histograms_mean))
                histogram_index.append((frame_index, histogram, [histogram]))
            else:
                previous_histograms.append(histogram)


    def _phash_frame_index(self, phash_index, frame, frame_index):
        hash = phash(frame)
        if len(phash_index) == 0:
            phash_index.append((frame_index, hash, [hash]))
        else:
            previous_frame_index, previous_hash, previous_hashes = phash_index[-1]
            if phash_dissimilarity(hash, previous_hash) > PHASH_MAX_DISSIMILARITY:
                previous_hashes_mean = np.array([item * 1.0 for item in previous_hashes]).mean(axis=0) > 0.5
                del phash_index[-1]
                phash_index.append((previous_frame_index, previous_hash, previous_hashes_mean))
                phash_index.append((frame_index, hash, [hash]))
            else:
                previous_hashes.append(hash)

    def index(self):
        histogram_index = []
        phash_index = []
        for i, frame in enumerate(self.dataset):
            self._hist_frame_index(histogram_index, frame, i)
            self._phash_frame_index(phash_index, frame, i)
        histogram_index_cleaned = [
            FrameIndex(self.fname, i, "hist", mean_hist)
            for i, hist, mean_hist in histogram_index
        ]
        histogram_index_cleaned[-1].value = np.array(histogram_index_cleaned[-1].value).mean(axis=0)
        phash_index_cleaned = [
            FrameIndex(self.fname, i, "hash", mean_hash)
            for i, hash, mean_hash in phash_index
        ]
        phash_index_cleaned[-1].value = np.array(phash_index_cleaned[-1].value).mean(axis=0)
        return histogram_index_cleaned, phash_index_cleaned