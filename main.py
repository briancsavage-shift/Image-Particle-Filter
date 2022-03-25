import os
from simulation import Simulate


def main():
    env = Simulate()
    env.loadMap(os.path.join(os.getcwd(), "maps", "BayMap.png"))

    for _ in range(10):
        env.timestep()
    env.export()


if __name__ == "__main__":
    outputDir = os.path.join(os.getcwd(), "output")
    if os.path.isdir(outputDir):
        for filepath in os.listdir(outputDir):
            os.remove(os.path.join(outputDir, filepath))
    else:
        os.mkdir(outputDir)

    main()
