from __future__ import division
from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk, square
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
from matplotlib import pylab as plt
import numpy as np
from numpy import array
from skimage import measure
import cv2 as cv


class Loader:
    def __init__(self, filename):
        self.image = None
        self._procImage = None
        self.contours = None
        self.__load_image(filename)
        self.centers = None
        self.figures = None

    def __load_image(self, filename):
        self.image = cv.imread(filename)
        self._procImage = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        if self.__determine_background() > 127:
            self._procImage = 255 - self._procImage
        ret, self._procImage = cv.threshold(self._procImage, 127, 255, cv.THRESH_BINARY)
        self.contours, hierarchy = cv.findContours(self._procImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    def calc_centers(self):
        self.centers = np.zeros(
            (len(self.contours), 2))  # inicjalizacja tablicy do obliczania środków ciężkości konturów
        for i in range(len(self.contours)):
            self.centers[i] = (
                [sum(self.contours[i][:, 0, 0]) / len(self.contours[i]),
                 sum(self.contours[i][:, 0, 1]) / len(self.contours[i])])

    def create_figures(self):
        self.figures = []
        for i in range(len(self.contours)):
            tmp = []
            for j in range(len(self.contours[i])):
                tmp.append(sqrt((self.contours[i][j, 0, 0] - self.centers[i][0]) ** 2 + (
                            self.contours[i][j, 0, 1] - self.centers[i][1]) ** 2))
            tmp = np.array(tmp)
            self.figures.append(Figure(self.contours[i], Measures(tmp, std(tmp), mean(tmp))))

    def __determine_background(self):
        tempDict = {}
        for i in self._procImage:
            for j in i:
                if j not in tempDict.keys():
                    tempDict[j] = 1
                else:
                    tempDict[j] += 1
        val = [(v, k) for k, v in tempDict.items()]
        val.sort(reverse=True)
        return val[0][0]


    def classify_figures(self):

        classified = []

        while len(classified) < len(self.figures):
            tMax = 0
            iMax = -1
            for i in range(len(self.figures)):
                if self.figures[i].measure.avg > tMax and i not in classified:
                    tMax = self.figures[i].measure.avg
                    iMax = i
            self.figures[iMax].type = "plansza"

            maxX = 0
            maxY = 0
            minX = self.figures[iMax].contour[0, 0, 0]
            minY = self.figures[iMax].contour[0, 0, 1]

            if iMax != -1:
                classified.append(iMax)

            for vert in self.figures[iMax].contour:
                if vert[0, 1] > maxX:
                    maxX = vert[0, 1]
                if vert[0, 1] < minX:
                    minX = vert[0, 1]
                if vert[0, 0] > maxY:
                    maxY = vert[0, 0]
                if vert[0, 0] < minY:
                    minY = vert[0, 0]

            within = []
            tMax1 = 0
            iMax1 = -1

            for i in range(len(self.centers)):
                if (minX < self.centers[i][0] < maxX) and (minY < self.centers[i][1] < maxY) and i not in classified:
                    within.append(i)
                    if self.figures[i].measure.avg > tMax1:
                        tMax1 = self.figures[i].measure.avg
                        iMax1 = i

            if iMax1 != -1 and abs(tMax - tMax1) < 0.15 * tMax:
                self.figures[iMax1].type = "plansza"
                within.remove(iMax1)
                classified.append(iMax1)

            tMax1 = 0
            iMax1 = -1
            tMin = -1

            for i in within:
                if self.figures[i].measure.avg > tMax1:
                    tmpradius = sqrt((self.centers[iMax][0] - self.centers[i][0])**2 + (self.centers[iMax][1] - self.centers[i][1])**2)
                    if tMin == -1 or tmpradius < tMin:
                        tMax1 = self.figures[i].measure.avg
                        iMax1 = i
                        tMin = tmpradius

            self.figures[iMax1].type = "srodek"
            within.remove(iMax1)
            classified.append(iMax1)

            sel_stdev = np.array([self.figures[i].measure.stdev for i in within])
            m_stdev = mean(sel_stdev)

            for i in within:
                if self.figures[i].measure.stdev <= m_stdev:
                    self.figures[i].type = "kolko"
                else:
                    self.figures[i].type = "krzyzyk"
                classified.append(i)


class Measures:
    def __init__(self, r, st, av):
        self.radius = r
        self.avg = av
        self.stdev = st


class Figure:
    def __init__(self, con, meas):
        self.contour = con
        self.measure = meas
        self.type = None


test = Loader('4.jpg')
test.calc_centers()
test.create_figures()
test.classify_figures()

plansza = io.imread('2.png', as_gray=True)  # ez wczytywanie
plansza = filters.sobel(plansza)

contours = measure.find_contours(plansza, 0)  # pyk kontury

fig, ax = plt.subplots()
ax.imshow(test.image, cmap=plt.cm.gray)

center = np.zeros((len(contours), 2))  # inicjalizacja tablicy do obliczania środków ciężkości konturów
# types = []

for i in range(len(contours)):
    center[i] = ([sum(contours[i][:, 1]) / len(contours[i]), sum(contours[i][:, 0]) / len(contours[i])])

delete = []  # lista konturów do usunięcia

for i in range(len(contours)):
    for j in range(0, i):
        if all(abs(center[j] - center[i]) <= (5, 5)):  # usuwamy kontury, których środki znajdują się zbyt blisko siebie
            delete.append(i)
            break
ct1 = []
cen1 = []
for i in range(len(center)):  # przepisanie list po usunięciu nadmiarowych konturów
    if i not in delete:
        ct1.append(contours[i])
        cen1.append(center[i])

contours = ct1
center = cen1

radius = []

figures = []

for i in range(len(contours)):
    tmp = []
    for j in range(len(contours[i])):
        tmp.append(sqrt((contours[i][j, 1] - center[i][0]) ** 2 + (contours[i][j, 0] - center[i][1]) ** 2))
    radius.append(tmp)

for i in range(len(radius)):
    tmp = np.array(radius[i])
    figures.append(Figure(contours[i], Measures(tmp, std(tmp), mean(tmp))))
col = 'r'
for i in range(0, len(test.figures)):  # rysowanie
    if test.figures[i].type == 'plansza':
        col = 'r'
    elif test.figures[i].type == 'kolko':
        col = 'g'
    elif test.figures[i].type == 'krzyzyk':
        col = 'b'
    else:
        col = 'y'
    ax.plot(test.figures[i].contour[:,0, 0], test.figures[i].contour[:,0, 1], linewidth=2, color=col)
    ax.plot(test.centers[i][0], test.centers[i][1], 'r+')

plt.savefig("test1.png")
