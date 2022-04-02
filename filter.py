import random
import numpy as np
import cv2
from simulation import Simulate


class ParticleFilter:
    def __init__(self, maxGenSize: int=100, simulation: Simulate = None):
        self.simulation = simulation
        self.maxGenSize = maxGenSize
        

    def sense(self, rounds: int) -> [(np.ndarray, np.ndarray, np.ndarray)]:
        
        bestGuess = []
        images = []
        points = self.generatePoints()
        for _ in range(rounds):
            
            weights = self.weightedSampling(points)
            imageR1 = self.drawPoints(weights, 
                                    points, 
                                    self.simulation.reference.copy())
            
            maxWX, maxWY = points[np.argmax(weights)]
            imageR1 = self.drawMarker(maxWX, maxWY, imageR1)
            bestGuess.append((maxWX, maxWY))
            
            sampled = []
            for r, (X, Y) in zip(weights, points):
                sampled.extend(self.resample(X, Y, r))
                
            nPoints = random.choices(sampled, k=self.maxGenSize) # All of same
                                                                 # weights
            
            eqSizes = [3] * len(nPoints)
            imageR2 = self.drawPoints(eqSizes,
                                      sampled, 
                                      self.simulation.reference.copy())

            moved = self.movePoints(nPoints)
            points = nPoints
            imageM1 = self.drawMoves(nPoints, 
                                     moved, 
                                     self.simulation.reference.copy())
            images.append((imageR1, imageR2, imageM1))
            

        return bestGuess, images

    def drawMoves(self, 
                  before: [(int, int)], 
                  after: [(int, int)], 
                  image: np.ndarray) -> np.ndarray:
        
        for (b, a) in zip(before, after):
            nB = self.simulation.convertCoordinates(b[0], b[1])
            nA = self.simulation.convertCoordinates(a[0], a[1])
            image = cv2.line(image, nB, nA, (128, 128, 30), 2)
        return image

    def generatePoints(self) -> [(int, int)]:
        """

        """
        points = []
        for _ in range(self.maxGenSize):
            rX = np.random.uniform(self.simulation.width // -2, 
                                   self.simulation.width // 2)
            rY = np.random.uniform(self.simulation.height // -2, 
                                   self.simulation.height // 2)
            points.append((int(rX), int(rY)))
        return points

    def weightedSampling(self, points: [(int, int)]) -> [int]:
        weights = []
        est = self.simulation.trueView()
        for (X, Y) in points:
            ref = self.simulation.getDroneView(X, Y)
            scr = self.similarityHeuristic(ref, est)
            radius = int(scr * self.maxGenSize)
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

        histRef = cv2.calcHist(ref, [0], None, [256], [0, 256])
        histExp = cv2.calcHist(exp, [0], None, [256], [0, 256])
        MSE = np.power((histRef - histExp), 2).mean()
        return np.divide(1, MSE) if MSE != 0.0 else 1.0

    def similarityComponentAnalysis(self, ref: np.ndarray, exp: np.ndarray) -> float:
        pass

    def similarityML(self, ref: np.ndarray, exp: np.ndarray) -> float:
        pass

    def movePoints(self, points: [(int, int)]) -> [(int, int)]:
        (dX, dY) = self.simulation.dXY[-1]
        mp = []
        for (X, Y) in points:
            cX = X + int(self.simulation.ppu * dX)
            cY = Y + int(self.simulation.ppu * dY)
            mp.append((cX, cY))
        return mp

    def drawPoints(self,
                   weights: [float],
                   points: [(int, int)],
                   image: np.ndarray) -> None:

        for w, (X, Y) in zip(weights, points):
            (nX, nY) = self.simulation.convertCoordinates(X, Y)
            image = cv2.circle(image, (nX, nY), w, (255, 255, 255), -1)
        return image

    def drawMarker(self, 
                   X: int, 
                   Y: int, 
                   image: np.ndarray) -> None:
        (nX, nY) = self.simulation.convertCoordinates(X, Y)
        image = cv2.circle(image, (nX, nY), 5, (0, 0, 255), -1)
        return image