import numpy as np
import cv2
from simulation import Simulate

class ParticleFilter:
    def __init__(self, sampleSize: int=250, simulation: Simulate=None):
        self.sampleSize = sampleSize
        self.simulation = simulation
        self.filteringRounds = 2

    def sense(self, estX, estY):
        images = []
        points = self.generatePoints()
        
        for _ in range(self.filteringRounds):
            
            weights = self.weightedSampling(points)
            image = self.drawPoints(weights, points, self.simulation.reference.copy())
            images.append(image)
        
            sampled = []
            for r, (X, Y) in zip(weights, points):
                sampled.extend(self.resample(X, Y, r))  
            points = sampled
            wMoves = self.movePoints(sampled)

        
        return images
        
        

    def generatePoints(self) -> [(int, int)]:
        """
        
        """
        points = []
        for _ in range(self.sampleSize):
            rX = np.random.uniform(self.simulation.width // -2, self.simulation.width // 2)
            rY = np.random.uniform(self.simulation.height // -2, self.simulation.height // 2)
            points.append((int(rX), int(rY)))
        return points

    def weightedSampling(self, points: [(int, int)]) -> [int]:
        weights = []
        est = self.simulation.estimatedView()
        for (X, Y) in points:
            ref = self.simulation.getDroneView(X, Y)
            scr = self.similarityHeuristic(ref, est)
            radius = int(scr * 100)
            weights.append(radius)            
        return weights
    
    def resample(self, cX: int, cY: int, radius: int) -> [(int, int)]:
        newPoints = []
        for _ in range(radius):
            aX = np.random.uniform(-radius, radius)
            aY = np.random.uniform(-radius, radius)
            newPoints.append((int(cX + aX), int(cY + aY)))
        return newPoints
    

    def similarityHeuristic(self, ref: np.ndarray, exp: np.ndarray) -> float:
        """

        """
        if ref.shape != exp.shape:
            return 0.0
        
        
        histRef = cv2.calcHist(ref,[0],None,[256],[0,256])
        histExp = cv2.calcHist(exp,[0],None,[256],[0,256])
        
        MSE = np.power((histRef - histExp), 2).mean()
        return np.divide(1, MSE) if MSE != 0.0 else 1.0
    
        
            
    def similarityComponentAnalysis(self, ref: np.ndarray, exp: np.ndarray) -> float:
        pass

    def similarityML(self, ref: np.ndarray, exp: np.ndarray) -> float:
        pass


    def movePoints(self, points: [(int, int)]) -> [(int, int)]:
        (dX, dY) = self.simulation.dXY[-1]
        points = []
        for (X, Y) in points:
            cX = int(X + (self.simulation.ppu * dX))
            cY = int(Y + (self.simulation.ppu * dY))
            points.append(cX, cY)
        return points
    
    
    def drawPoints(self, 
                   weights: [float], 
                   points: [(int, int)], 
                   image: np.ndarray) -> None:
        
        for w, (X, Y) in zip(weights, points):
            (nX, nY) = self.simulation.convertCoordinates(X, Y)
            image = cv2.circle(image, (nX, nY), w, (255, 255, 255), -1)
        return image