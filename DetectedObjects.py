import numpy as np

class DetectedObjects:
    def __init__(self, centroid: tuple[int, int], contour: np.ndarray):
        self.centroid = centroid
        self.contour = contour

    def __repr__(self):
        return f"Barrel(centroid={self.centroid}, contour_shape={self.contour.shape})"




