import matplotlib.pyplot as plt
import os
from collections import defaultdict

class MatplotlibVisualize():
    def __init__(self, enable=True):
        '''
            Initialize visualization using matplotlib.
        '''
        self.is_enabled = enable
        self.data = defaultdict(list)  # Store all points for each key-line pair

    def linePlot(self, x, y, key, line_name, xlabel="Iterations"):
        '''
            Simulate Visdom's line plotting using matplotlib.
            Args:
                x : Scalar -> X-coordinate on plot
                y : Scalar -> Value at x
                key : Name of plot/graph
                line_name : Name of line within plot/graph
                xlabel : Label for x-axis
        '''
        if not self.is_enabled:
            print(123)
            return

        self.data[(key, line_name)].append((x, y))
        plt.figure(key)
        plt.clf()
        for (k, ln), points in self.data.items():
            if k == key:
                xs, ys = zip(*points)
                plt.plot(xs, ys, label=ln)
        plt.xlabel(xlabel)
        plt.ylabel(key)
        plt.title(key)
        plt.legend()
        plt.grid(True)
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{key}.png")
        plt.close()

    def showText(self, text, key):
        '''
            Simulate text display by writing to a text file.
        '''
        if self.is_enabled:
            os.makedirs("logs", exist_ok=True)
            with open(f"logs/{key}.txt", "w") as f:
                f.write(text)

    def addText(self, text):
        '''
            Append unnamed text to a log file.
        '''
        if self.is_enabled:
            os.makedirs("logs", exist_ok=True)
            with open("logs/unnamed.txt", "a") as f:
                f.write(text + "\n")

