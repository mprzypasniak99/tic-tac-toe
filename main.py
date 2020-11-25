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
        self.figures = None
        self.games = None

    def getproc(self):
        return self._procImage

    def __load_image(self, filename):
        self.image = cv.imread(filename)
        self.image = cv.resize(self.image, (500, 500))
        self.__preprocess()
        self.contours, hierarchy = cv.findContours(self._procImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    def __preprocess(self):
        kernel = np.ones((4, 4), np.uint8)
        self._procImage = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self._procImage = cv.bilateralFilter(self._procImage, 9, 5, 5)
        self._procImage = cv.Canny(self._procImage, 20, 50)
        self._procImage = cv.morphologyEx(self._procImage, cv.MORPH_CLOSE, kernel)
        ret, self._procImage = cv.threshold(self._procImage, 127, 255, cv.THRESH_BINARY)

    def __filter_figures(self):
        i = 0
        while i < len(self.figures):
            if cv.contourArea(self.figures[i].convHull) < 100:
               self.figures.pop(i)
            else:
                i += 1

    def __find_lines(self):
        linesP = cv.HoughLinesP(self._procImage, 1, np.pi / 240, 50, None, 60, 15)
        lines = []
        if len(linesP != 0):
            for i in range(len(linesP)):
                l = linesP[i]
                lines.append(Line([(l[0][0], l[0][1]), (l[0][2], l[0][3])]))
        return lines

    def create_figures(self):
        self.figures = []
        for i in range(len(self.contours)):

            center = (
                [sum(self.contours[i][:, 0, 0]) / len(self.contours[i]),
                 sum(self.contours[i][:, 0, 1]) / len(self.contours[i])])
            tmpRad = []

            hull = cv.convexHull(self.contours[i])

            for j in range(len(self.contours[i])):
                tmpRad.append(sqrt((self.contours[i][j, 0, 0] - center[0]) ** 2 + (
                            self.contours[i][j, 0, 1] - center[1]) ** 2))
            tmpRad = np.array(tmpRad)
            self.figures.append(Figure(self.contours[i], center, hull, Measures(tmpRad, std(tmpRad), mean(tmpRad))))
        self.__filter_figures()

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

            for i in range(len(self.figures)):
                if (minX < self.figures[i].center[0] < maxX) and (minY < self.figures[i].center[1] < maxY) and i not in classified:
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
                    tmpradius = sqrt((self.figures[iMax].center[0] - self.figures[i].center[0])**2
                                     + (self.figures[iMax].center[1] - self.figures[i].center[1])**2)
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

    def classify_figures2(self):
        for i in range(len(self.figures)):
            hull_area = cv.contourArea(self.figures[i].convHull)
            con_area = cv.contourArea(self.figures[i].contour)
            if con_area / hull_area < 0.5:
                self.figures[i].type = 'krzyzyk'
            else:
                self.figures[i].type = 'kolko'

    def find_games(self):
        self.games = []
        assigned = []
        lines = self.__find_lines()

        while len(assigned) < len(self.figures):
            maxi = 0
            while maxi in assigned:
                maxi += 1
            for i in range(len(self.figures)):
                if (self.figures[i].measure.avg > self.figures[maxi].measure.avg and i not in assigned and
                        self.figures[i].type == 'krzyzyk'):
                    maxi = i
            assigned.append(maxi)

            maxx = 0
            maxy = 0
            minx = self.figures[maxi].contour[0, 0, 0]
            miny = self.figures[maxi].contour[0, 0, 1]

            for vert in self.figures[maxi].contour:
                if vert[0, 1] > maxy:
                    maxy = vert[0, 1]
                if vert[0, 1] < miny:
                    miny = vert[0, 1]
                if vert[0, 0] > maxx:
                    maxx = vert[0, 0]
                if vert[0, 0] < minx:
                    minx = vert[0, 0]

            within = []
            lines_within = []
            for i in range(len(self.figures)):
                if (minx < self.figures[i].center[0] < maxx and miny < self.figures[i].center[1] < maxy and
                        i not in assigned):
                    within.append(i)
                    assigned.append(i)

            for i in range(len(lines)):
                if ((minx < lines[i].points[0][0] < maxx and miny < lines[i].points[0][1] < maxy) and
                        (minx < lines[i].points[0][0] < maxx and miny < lines[i].points[0][1] < maxy)):
                    lines_within.append(lines[i])

            if len(within) != 0:
                self.games.append(Game([(maxx, maxy), (minx, miny)], within, lines_within))

        #for g in self.games:
         #   for l in g.lines:
          #      cv.line(self.image, (l.points[0][0], l.points[0][1]),
           #             (l.points[1][0], l.points[1][1]), (0, 0, 255), 3, cv.LINE_AA)

       # cv.imshow('window', self.image)
        #cv.waitKey(0)

    def classify_games(self):
        for game in self.games:
            len_x = abs(game.border[0][0] - game.border[1][0]) // 3 # długość x jednego pola
            len_y = abs(game.border[0][1] - game.border[1][1]) // 3 # długość y jednego pola
            for index in range(9):
                lx = game.border[1][0] + len_x * (index % 3) # lewa granica pola
                rx = game.border[1][0] + len_x * ((index % 3) + 1)  # prawa granica pola
                dy = game.border[1][1] + len_y * (index // 3) # górna granica pola
                uy = game.border[1][1] + len_y * ((index // 3) + 1) # dolna granica pola

                in_field = []
                for j in game.figures:
                    for vert in self.figures[j].contour:
                        if lx < vert[0, 0] < rx and dy < vert[0, 1] < uy:
                            in_field.append(j)
                            break

                if len(in_field) == 0:
                    self.image = cv.putText(self.image, '-', (lx + int(0.5*len_x), uy - int(0.5 * len_y)),
                                            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                else:
                    maxi = in_field[0]
                    for index in in_field:
                        if self.figures[index].measure.avg > self.figures[maxi].measure.avg:
                            maxi = index
                    if self.figures[maxi].type == 'kolko':
                        self.image = cv.putText(self.image, 'O', (lx + int(0.5*len_x), uy - int(0.5 * len_y)),
                                                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                    else:
                        self.image = cv.putText(self.image, 'X', (lx + int(0.5*len_x), uy - int(0.5 * len_y)),
                                                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)


class Measures:
    def __init__(self, r, st, av):
        self.radius = r
        self.avg = av
        self.stdev = st


class Figure:
    def __init__(self, con, cen, hull, meas):
        self.contour = con
        self.center = cen
        self.convHull = hull
        self.measure = meas
        self.type = None

class Line:
    def __init__(self, points: list):
        self.points = points
        try:
            self.dirx = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0]) #y1-y2/x1-x2
            self.addx = points[0][1] - self.dirx * points[0][0]
            self.diry = None
            self.addy = None
        except ZeroDivisionError:
            self.dirx = None
            self.addx = None
            self.diry = (points[0][0] - points[1][0]) / (points[0][1] - points[1][1])
            self.addy = points[0][0] - self.diry * points[0][1]
        self.len = sqrt((points[0][1] - points[1][1])**2 + (points[0][0] - points[1][0])**2)


class Game:
    def __init__(self, border, fig, l: list):
        self.border = border  #punkty określające granice planszy
        self.figures = fig  #indeksy figur w klasie "Loader"
        self.lines = l
        self.__simplify_lines()

    def __simplify_lines(self):
        length = np.array([])
        indexes = list(range(len(self.lines)))
        for i in indexes:
            j = i+1
            while j < len(indexes):
                pi = self.lines[i]
                pj = self.lines[j]
                to_del = -1

                diffa = 5
                diffb = 40

                if (j != i and ((pi.dirx is not None and pj.dirx is not None and abs(pi.dirx - pj.dirx) < diffa
                                 and abs(pi.addx - pj.addx) < diffb))):
                    ind = [(0, 0), (1, 0), (1, 1), (0, 1)]
                    dist = [sqrt((pi.points[i][0] - pj.points[j][0]) ** 2 + (pi.points[i][1] - pj.points[j][1]) ** 2)
                            for i,j in ind]
                    p1, p2 = ind[dist.index(max(dist))]

                    self.lines[i] = Line([pi.points[p1], pj.points[p2]])
                    to_del = j

                if to_del != -1:
                    self.lines.pop(to_del)
                    indexes.pop()
                else:
                    j += 1
            length = np.append(length, self.lines[i].len)
        avg = np.mean(length)
        i = 0
        while i < len(self.lines):
            if self.lines[i].len < avg:
                self.lines.pop(i)
            else:
                i += 1
#files = ["21", "23", "25", "26", "27", "29", "30", "32", "33", "34", "35", "36", "37"]
files = ['3', '4']
for f in files:
    test = Loader("resources/"+f+".jpg")
    test.create_figures()
    test.classify_figures2()
    test.find_games()
    test.classify_games()

    im = test.image
    for i in range(len(test.games)):
        im2 = cv.rectangle(im, test.games[i].border[0], test.games[i].border[1], (0, 255, 0), 2)

    cv.imwrite('complete/'+f+'.png', im2)


fig, ax = plt.subplots()
ax.imshow(test.image, cmap=plt.cm.gray)

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
    ax.plot(test.figures[i].contour[:, 0, 0], test.figures[i].contour[:, 0, 1], linewidth=2, color=col)
    #ax.plot(test.figures[i].convHull[:, 0, 0], test.figures[i].convHull[:, 0, 1], linewidth=1, color='m')
    ax.plot(test.figures[i].center[0], test.figures[i].center[1], 'r+')

im = test.image


#for i in range(len(test.contours)):
 #   im = cv.drawContours(im, test.contours, i, (255, 0, 0), 3)



cv.imshow('window', im)

cv.waitKey(0)

plt.savefig("test1.png")
