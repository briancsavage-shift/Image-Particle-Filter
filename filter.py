import numpy as np
from simulation import Simulate

class ParticleFilter:
    def __init__(self, sampleSize: int=100, simulation: Simulate=None):
        self.sampleSize = sampleSize
        self.simulation = simulation

    def generatePoints(self):
        """
        
        """
        points = []
        for _ in range(self.sampleSize):
            rX = np.random.uniform(self.simulation.height)
            rY = np.random.uniform(self.simulation.width)
            points.append((int(rX), int(rY)))
        return points

    def similarityHeuristic(self, ref: np.ndarray, exp: np.ndarray) -> float:
        """

        """
        return 1.0 - np.tanh(np.sum(np.absolute(np.subtract(ref, exp))))

    def similarityComponentAnalysis(self, ref: np.ndarray, exp: np.ndarray) -> float:
        pass

    def similarityML(self, ref: np.ndarray, exp: np.ndarray) -> float:
        pass

    def weightedSampling(self, points: [(int, int)]) -> [float]:
        pass

    def movePoints(self, points: [(int, int)], dX: float, dY: float) -> [(int, int)]:
        return points