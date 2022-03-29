import numpy as np
import cv2
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
        histRef = cv2.calcHist(ref,[0],None,[256],[0,256])
        histExp = cv2.calcHist(exp,[0],None,[256],[0,256])
        
        MSE = np.power((histRef - histExp), 2).mean()
        return np.divide(1, MSE) if MSE != 0.0 else 1.0
    
        
            
    def similarityComponentAnalysis(self, ref: np.ndarray, exp: np.ndarray) -> float:
        pass

    def similarityML(self, ref: np.ndarray, exp: np.ndarray) -> float:
        pass

    def weightedSampling(self, points: [(int, int)]) -> [float]:
        pass

    def movePoints(self, points: [(int, int)], dX: float, dY: float) -> [(int, int)]:
        return points