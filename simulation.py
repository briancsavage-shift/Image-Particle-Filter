import os
import numpy as np
import cv2


class Simulate:
    def __init__(self, observedPixels: int = 50, pixelsPerUnit: int = 50):
        self.X = 0  # Starting X
        self.Y = 0  # Starting Y
        self.sigmaMovement = 0.5  # Standard deviation of movement

        self.dXY = []
        self.pos = [(self.X, self.Y)]
        self.ppu = pixelsPerUnit
        self.observedPixels = observedPixels

        self.indicatorColor = (0, 0, 255)  # BGR format for OpenCV
        self.indicatorRadius = 5

    def timestep(self) -> None:
        """

        """
        dX = np.random.uniform(-1, 1)
        dY = np.sqrt(np.subtract(1, np.power(dX, 2))) * \
            np.random.choice([-1, 1])

        while not self.inBounds(dX, dY):
            dX = np.random.uniform(-1, 1)
            dY = np.sqrt(np.subtract(1, np.power(dX, 2))) * \
                np.random.choice([-1, 1])
        print(f"dX: {dX}, dY: {dY}, constraint: {dX ** 2 + dY ** 2}")

        self.X += int(dX * self.ppu)
        self.Y += int(dY * self.ppu)

        self.X += int(np.random.uniform(0, self.sigmaMovement ** 2) * self.ppu)
        self.Y += int(np.random.uniform(0, self.sigmaMovement ** 2) * self.ppu)

        self.dXY.append((dX, dY))
        self.pos.append((self.X, self.Y))
        print(f"X: {self.X}, Y: {self.Y}")

        self.droneView = self.getDroneView(self.X, self.Y)

    def setViewSize(self, size: int = 25) -> None:
        self.observedPixels = size

    def getDroneRef(self) -> np.ndarray:
        pX = int(sum([x * self.ppu for (x, y) in self.dXY]))
        pY = int(sum([y * self.ppu for (x, y) in self.dXY]))
        return self.getDroneView(pX, pY)

    def getDroneEnv(self) -> np.ndarray:
        return self.getDroneView(self.X, self.Y)

    def getDroneView(self, X: int, Y: int) -> np.ndarray:
        """

        """
        droneView = self.environment.copy()
        X, Y = self.convertCoordinates(X, Y)
        return droneView[Y - (self.observedPixels // 2): Y + (self.observedPixels // 2),
                         X - (self.observedPixels // 2): X + (self.observedPixels // 2)]

    def inBounds(self, dX: float, dY: float) -> bool:
        """

        """
        nX = self.X + int(dX * self.ppu)
        nY = self.Y + int(dY * self.ppu)

        (nX, nY) = self.convertCoordinates(nX, nY)
        print(
            f"Checking nX: {nX}, nY: {nY} against width: {self.width}, height: {self.height}")
        return (nX >= 0 and nX < self.width) and (nY >= 0 and nY < self.height)

    def loadMap(self, filepath: str = None) -> None:
        """


        """
        if os.path.isfile(filepath):
            self.environment = cv2.imread(filepath)
            self.height, self.width = self.environment.shape[:2]
            self.reference = np.zeros((self.height, self.width, 3), np.uint8)

            for i in range(0, self.height):
                for j in range(0, self.width):
                    options = {
                        (True, True): (0, 255, 0),
                        (True, False): (255, 255, 255),
                        (False, True): (255, 255, 255),
                        (False, False): (0, 0, 0),
                    }

                    # x, y = self.convertCoordinates(i, j)
                    case = (i % self.ppu == 0, j % self.ppu == 0)
                    self.reference[i, j] = options[case]

        else:
            raise FileNotFoundError("Map file not found.")

    def convertCoordinates(self, X: int, Y: int) -> (int, int):
        """

        """
        return (X + (self.width // 2), Y + (self.height // 2))

    def export(self) -> None:
        """
            @ Does
            - Generates a reference image for a particular position of the map.
            - This is used to compare the output of the simulation env to the 
              ref image.

            @ Notes
            - Saves ref and env PNG images within the output directory

        """
        reference = self.reference.copy()
        environment = self.environment.copy()
        for i, (X, Y) in enumerate(self.pos):
            if i > 0:
                cv2.line(reference,
                         self.convertCoordinates(
                             self.pos[i - 1][0], self.pos[i - 1][1]),
                         self.convertCoordinates(X, Y),
                         (255, 255, 255), self.indicatorRadius // 2)
                cv2.line(environment,
                         self.convertCoordinates(
                             self.pos[i - 1][0], self.pos[i - 1][1]),
                         self.convertCoordinates(X, Y),
                         (255, 255, 255), self.indicatorRadius // 2)

        for ((X, Y), alpha) in zip(self.pos, np.linspace(0.5, 1, len(self.pos))):
            indicatorColor = [int(c * alpha) for c in self.indicatorColor]
            cv2.circle(reference,
                       self.convertCoordinates(X, Y),
                       self.indicatorRadius + 3, (255, 255, 255), -1)
            cv2.circle(environment,
                       self.convertCoordinates(X, Y),
                       self.indicatorRadius + 3, (255, 255, 255), -1)
            cv2.circle(reference,
                       self.convertCoordinates(X, Y),
                       self.indicatorRadius, indicatorColor, -1)
            cv2.circle(environment,
                       self.convertCoordinates(X, Y),
                       self.indicatorRadius, indicatorColor, -1)

        aX = int(sum([x * self.ppu for (x, y) in self.dXY]))
        aY = int(sum([y * self.ppu for (x, y) in self.dXY]))

        cv2.circle(reference,
                   self.convertCoordinates(aX, aY),
                   self.indicatorRadius, (0, 255, 0), -1)
        cv2.circle(environment,
                   self.convertCoordinates(aX, aY),
                   self.indicatorRadius, (0, 255, 0), -1)

        savepath = os.path.join(os.getcwd(), "output")
        cv2.imwrite(os.path.join(savepath, "ref.png"), reference)
        cv2.imwrite(os.path.join(savepath, "env.png"), environment)
        return (reference, environment)

    def reset(self):
        self.pos = [(0, 0)]
        self.dXY = []
