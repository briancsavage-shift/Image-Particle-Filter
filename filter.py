import random
import numpy as np
import cv2

from typing import List, Tuple
from simulation import Simulate
from extractor import PerspectiveSimularity

class ParticleFilter:
    def __init__(self, maxGenSize: int = 100, simulation: Simulate = None):
        self.simulation = simulation
        self.maxGenSize = maxGenSize
        self.fextractor = PerspectiveSimularity()

    def senseWithHeuristic(self, rounds: int) -> Tuple[List[Tuple[int, int]],
                                                       List[float],
                                                       List[Tuple[np.ndarray,
                                                                  np.ndarray,
                                                                  np.ndarray]]]:
        bestXY, images, scores = [], [], []
        points = self.generatePoints()
        for _ in range(rounds):
            weights = self.weightedSampling(points)     # Weights sampled points
            imageR1 = self.drawPoints(weights,
                                      points,
                                      self.simulation.reference.copy())

            # Best estimate after weighted sampling
            (maxWX, maxWY) = points[np.argmax(weights)]
            imageR1 = self.drawMarker(maxWX, maxWY, imageR1)
            bestXY.append((maxWX, maxWY))

            sampled = []
            for r, (X, Y) in zip(weights, points):
                sampled.extend(self.resample(X, Y, r))   # Resampling on weights

            nPoints = random.choices(sampled, k=self.maxGenSize)  # Equal weight
            eqSizes = [3] * len(nPoints)          # Scale for rendering purposes
            imageR2 = self.drawPoints(eqSizes,
                                      sampled,
                                      self.simulation.reference.copy())

            moved = self.movePoints(nPoints)      # Shift points based on dX, dY
            imageM1 = self.drawMoves(nPoints,
                                     moved,
                                     self.simulation.reference.copy())
            points = nPoints
            images.append((imageR1, imageR2, imageM1))
            scores.append(self.score(positionEstimate=(maxWX, maxWY)))
        return (bestXY, scores, images)

    def senseWithLearning(self, rounds: int) -> Tuple[List[Tuple[int, int]],
                                                      List[np.ndarray],
                                                      List[np.ndarray]]:
        trainingData, trainingView = [], []
        for _ in range(rounds):
            (validPoints, validViews) = self.validViews(self.generatePoints())
            trainingData += validPoints
            trainingView += validViews

        distY = []
        for i, (x, y) in enumerate(trainingData):
            distY.append(self.score(positionEstimate=(x, y)))
        self.fextractor.train(trainingView, distY)

        bestXY, images, scores = [], [], []
        points, views = self.validViews(self.generatePoints())
        for _ in range(rounds):
            weights = self.learnedSampling(points)
            imageR1 = self.drawPoints(weights,
                                      points,
                                      self.simulation.reference.copy())

            (maxWX, maxWY) = points[np.argmax(weights)]
            imageR1 = self.drawMarker(maxWX, maxWY, imageR1)
            bestXY.append((maxWX, maxWY))

            sampled = []
            for r, (X, Y) in zip(weights, points):
                sampled.extend(self.resample(X, Y, r))

            points, _ = self.validViews(self.generatePoints())
            sampled, views = self.validViews(sampled + points)

            if len(sampled) > self.maxGenSize:
                nPoints = random.choices(sampled, k=self.maxGenSize)
            else:
                nPoints = sampled

            eqSizes = [8] * len(nPoints)
            imageR2 = self.drawPoints(eqSizes,
                                      sampled,
                                      self.simulation.reference.copy())

            moved = self.movePoints(nPoints)
            imageM1 = self.drawMoves(nPoints,
                                     moved,
                                     self.simulation.reference.copy())
            points = nPoints
            images.append((imageR1, imageR2, imageM1))
            scores.append(self.score(positionEstimate=(maxWX, maxWY)))
        return (bestXY, scores, images)

    def drawMoves(self,
                  before: List[Tuple[int, int]],
                  after: List[Tuple[int, int]],
                  image: np.ndarray) -> np.ndarray:

        for (b, a) in zip(before, after):
            nB = self.simulation.convertCoordinates(b[0], b[1])
            nA = self.simulation.convertCoordinates(a[0], a[1])
            image = cv2.line(image, nB, nA, (128, 128, 30), 2)
        return image

    def generatePoints(self) -> List[Tuple[int, int]]:
        points = []
        for _ in range(self.maxGenSize):
            rX = np.random.uniform(self.simulation.width // -2,
                                   self.simulation.width // 2)
            rY = np.random.uniform(self.simulation.height // -2,
                                   self.simulation.height // 2)
            points.append((int(rX), int(rY)))
        return points

    def validViews(self,
                   points: List[Tuple[int, int]]) -> Tuple[List[Tuple[int,
                                                                      int]],
                                                           List[np.ndarray]]:
        valid_points, valid_views = [], []
        for (x, y) in points:
            view = self.simulation.getDroneView(x, y)
            if view.shape[0] == self.simulation.observedPixels - 1 and \
               view.shape[1] == self.simulation.observedPixels - 1:
                valid_points.append((x, y))
                valid_views.append(view)
        return valid_points, valid_views

    def weightedSampling(self, points: List[Tuple[int, int]]) -> List[int]:
        weights = []
        est = self.simulation.trueView()
        for (X, Y) in points:
            ref = self.simulation.getDroneView(X, Y)
            wgt = self.similarityHeuristic(ref, est)
            radius = int(wgt * self.maxGenSize)
            weights.append(radius)
        return weights

    def learnedSampling(self, points: List[Tuple[int, int]]) -> List[int]:
        testViews = []
        for (X, Y) in points:
            testViews.append(self.simulation.getDroneView(X, Y))
        (weights, smaxWeights) = self.fextractor.predict(testViews)
        return [int(weight * 20) for weight in smaxWeights]

    def resample(self, cX: int, cY: int, radius: int) -> List[Tuple[int, int]]:
        newPoints = []
        for _ in range(radius):
            aX = np.random.uniform(-radius, radius)
            aY = np.random.uniform(-radius, radius)
            newPoints.append((int(cX + aX), int(cY + aY)))
        return newPoints

    def scoreSet(self, points: List[Tuple[int, int]]) -> List[float]:
        return [self.score(positionEstimate=p) for p in points]

    def score(self, positionEstimate: Tuple[int, int]) -> float:
        (trueX, trueY) = (self.simulation.X, self.simulation.Y)
        (eposX, eposY) = positionEstimate

        current = self.euclidean(trueX, trueY, eposX, eposY)
        maximum = self.euclidean(self.simulation.width,
                                 self.simulation.height, 0, 0)
        return 1.0 - (current / maximum)

    @staticmethod
    def euclidean(trueX: int, trueY: int, eposX: int, eposY: int) -> float:
        return np.sqrt((trueX - eposX) ** 2 + (trueY - eposY) ** 2)

    @staticmethod
    def similarityHeuristic(ref: np.ndarray, exp: np.ndarray) -> float:
        """
            Calculates the similarity between two images using a heuristic
            function.

            :param ref: (np.ndarray) perspective image from tested position
            :param exp: (np.ndarray) ground truth image from drone position
            :return (float): similarity value between 0 and 1
        """
        if ref.shape != exp.shape:
            return 0.0

        histRef = cv2.calcHist(ref, [0], None, [256], [0, 256])
        histExp = cv2.calcHist(exp, [0], None, [256], [0, 256])
        MSE = np.power((histRef - histExp), 2).mean()
        return np.divide(1, MSE) if MSE != 0.0 else 1.0

    @staticmethod
    def similarityML(ref: np.ndarray, exp: np.ndarray) -> float:
        """
            Calculates the similarity between two images using a learning based
            function.

            :param ref: (np.ndarray) perspective image from tested position
            :param exp: (np.ndarray) ground truth image from drone position
            :return (float): similarity value between 0 and 1
        """
        pass

    def movePoints(self, points: List[Tuple[int, int]]) -> List[Tuple[int,
                                                                      int]]:
        mp = []
        (dX, dY) = self.simulation.dXY[-1]
        for (X, Y) in points:
            cX = X + int(self.simulation.ppu * dX)
            cY = Y + int(self.simulation.ppu * dY)
            mp.append((cX, cY))
        return mp

    def drawPoints(self,
                   weights: List[float],
                   points: List[Tuple[int, int]],
                   image: np.ndarray) -> np.ndarray:

        for w, (X, Y) in zip(weights, points):
            (nX, nY) = self.simulation.convertCoordinates(X, Y)
            image = cv2.circle(image, (nX, nY), w, (255, 255, 255), -1)
        return image

    def drawMarker(self, X: int, Y: int, image: np.ndarray) -> None:
        (nX, nY) = self.simulation.convertCoordinates(X, Y)
        image = cv2.circle(image, (nX, nY), 5, (0, 0, 255), -1)
        return image
