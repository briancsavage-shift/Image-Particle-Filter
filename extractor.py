import numpy as np
import cv2

from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 y: int = None,
                 orientations: int = 9,
                 pixelsPerCell: (int, int) = (8, 8),
                 cellsPerBlock: (int, int) = (2, 2),
                 blockNorm: str = 'L2-Hys'):

        self.y = y
        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.blockNorm = blockNorm

    def fit(self, X: np.ndarray, y: float=None):
        return self

    def transform(self, images: [np.ndarray], y: float=None) -> [np.ndarray]:
        extractFeatures = lambda image : hog(image,
                                             orientations=self.orientations,
                                             pixels_per_cell=self.pixelsPerCell,
                                             cells_per_block=self.cellsPerBlock,
                                             block_norm=self.blockNorm)
        try:
            grayed = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
            features = np.array(list(map(extractFeatures, grayed)))
            return StandardScaler().fit_transform(features)
        except Exception as e:
            print(f"Error Raised: {e}")



class PerspectiveSimularity:
    def __init__(self):
        self.featureExtractor = FeatureExtractor(orientations=9,
                                                 pixelsPerCell=(8, 8),
                                                 cellsPerBlock=(2, 2))

    def train(self, X: [np.ndarray], y: [float]):
        self.regressor = SGDRegressor(loss='squared_loss',
                                      max_iter=1000,
                                      tol=1e-3)

        X = self.featureExtractor.fit_transform(X)
        self.regressor.fit(X, y)

    def predict(self, X: [np.ndarray]) -> [float]:
        X = self.featureExtractor.fit_transform(X)
        return self.regressor.predict(X)

    @staticmethod
    def score(reference: np.ndarray, expected: np.ndarray) -> float:
        pass
