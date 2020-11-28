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
        self.image = None  # field for storing raw loaded image
        self._procImage = None  # private field storing image prepared for analysis
        self.contours = None  # array storing contours found in image
        self.hierarchy = None  # matrix storing hierarchy of contours: [previous, next, first_child, parent] By Borubasz
        self.convex_hulls = None  # convex hulls of contours By Borubasz
        self.__load_image(filename)  # execute method loading image, preprocess it and find contours
        self.__get_convex_hulls()  # By Borubasz
        self.__filter_contours()  # delete small contours from the array
        self.figures = None  # array storing Figure objects created in the image
        self.games = None  # array storing Game objects found in the image

    def getproc(self):  # return preprocessed image - used for debugging
        return self._procImage

    def __load_image(self, filename):  # method for loading image and getting contours out of it
        self.image = cv.imread(filename)
        self.image = cv.resize(self.image, (500, 500))
        # resize image - gives better results when smaller image is used in preprocessing

        self.__preprocess()  # prepare image for analysis

        self.contours, self.hierarchy = cv.findContours(self._procImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # get contours out of the image

    def __get_convex_hulls(self):  # By Borubasz
        self.convex_hulls = []
        for i in self.contours:
            self.convex_hulls.append(cv.convexHull(i))

    def __detect_lines(self, image, secImage):  # By Borubasz
        lines = cv.HoughLines(image, 2, np.pi / 180, 50, None, 50, 10)
        strong_lines = []
        maxy, maxx = image.shape[:2]
        avg = (maxy+maxx)/2
        for i in lines:  # recalculating parameters for lines that have negative rho
            if i[0][0] < 0:
                i[0][0] *= -1.0
                i[0][1] -= np.pi

        for i in lines:  # finding four strong (that means they have the most points) and different lines
            chk = 0
            if len(strong_lines) == 0:
                strong_lines.append(i)
                continue
            for j in strong_lines:
                if (j[0][0] - 0.2*avg <= i[0][0] <= j[0][0] + 0.2*avg) and (
                        j[0][1] - np.pi / 18 <= i[0][1] <= j[0][1] + np.pi / 18):
                    chk += 1
            if chk == 0 and len(strong_lines) < 4:
                strong_lines.append(i)

        if strong_lines is not None:  # this fragment of code draw those strong lines
            for i in range(0, len(strong_lines)):
                rho = strong_lines[i][0][0]
                theta = strong_lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv.line(secImage, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)
        cv.imshow('window', secImage)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # let's calculate line equations
        lines_eq = []  # structure will look like this: [a, b, c]
        for i in range(0, len(strong_lines)):
            rho = strong_lines[i][0][0]
            theta = strong_lines[i][0][1]
            try:
                a = -1*math.cos(theta) / math.sin(theta)
                b = rho / math.sin(theta)
                c = -inf
            except ZeroDivisionError:
                a = -inf
                b = -inf
                c = rho
            lines_eq.append([a, b, c])

        lines_eq.sort(key=lambda x: -inf if x[0] == -inf else abs(x[0]))
        print(lines_eq)
        return lines_eq

    def __handle_game(self, minx: int, miny: int, maxx: int, maxy: int):  # By Borubasz
        game = self._procImage[miny:maxy, minx:maxx]  # cutting out founded field
        tmpGame = self.image[miny:maxy, minx:maxx]  # cutting out founded field
        print(minx, ' ', maxx, ' ', miny, ' ', maxy)
        cv.imshow('window', tmpGame)
        cv.waitKey(0)
        cv.destroyAllWindows()
        lines_eq = self.__detect_lines(game, tmpGame)
        if len(lines_eq) < 4:
            return
        points_of_interest = [[0, 0], [0, (maxx - minx)], [(maxy - miny), 0], [(maxy - miny), (maxx - minx)]]  # [y, x]
        for i in range(2):
            for j in range(2, 4):
                if lines_eq[i][0] == -inf:
                    x = lines_eq[i][2]
                    y = lines_eq[j][0]*x+lines_eq[j][1]
                else:
                    x = (lines_eq[i][1] - lines_eq[j][1])/(lines_eq[i][0] - lines_eq[j][0])
                    y = lines_eq[i][0]*x+lines_eq[i][1]
                    print(i, ' ', j, ' ', x, ' ', y)
                    points_of_interest.append([int(y), int(x)])
        print(len(points_of_interest))
        contours, hierarchy = cv.findContours(game, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        convex_hulls = []
        for i in contours:
            convex_hulls.append(cv.convexHull(i))
        for i in range(len(contours)):
            if hierarchy[0, i, 2] == -1:
                hull_area = cv.contourArea(convex_hulls[i])  # compute area of convex hull of the figure
                con_area = cv.contourArea(contours[i])  # compute area of the contour of the figure
                center = (int(sum(contours[i][:, 0, 0]) / len(contours[i])),
                          int(sum(contours[i][:, 0, 1]) / len(contours[i])))
                if con_area / hull_area > 0.5 and con_area > 0.001*(maxx-minx)*(maxy-miny):
                    tmpGame = cv.putText(tmpGame, 'O', center, cv.FONT_HERSHEY_SIMPLEX,
                                         1, (255, 0, 255), 3)
                elif con_area / hull_area <= 0.5 and con_area > 0.001*(maxx-minx)*(maxy-miny):
                    tmpGame = cv.putText(tmpGame, 'X', center, cv.FONT_HERSHEY_SIMPLEX,
                                         1, (255, 0, 255), 3)
        cv.imshow('window', tmpGame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_game(self):  # By Borubasz: this little function finds contour with biggest convex hull
        # if you would take a look at the photo with drawn convex hull you would find that convex hull pretty nicely
        # outlines game field
        assigned = []
        while len(assigned) < len(self.convex_hulls):
            ind = 0

            handicap = 10
            for i in range(1, len(self.convex_hulls)):
                if (i not in assigned and
                cv.contourArea(self.convex_hulls[ind]) < cv.contourArea(self.convex_hulls[i])):
                    ind = i
            # we take maxes and mins from hull to find rectangle including game
            maxx = max(self.convex_hulls[ind][:, 0, 0])
            maxy = max(self.convex_hulls[ind][:, 0, 1])
            minx = min(self.convex_hulls[ind][:, 0, 0])
            miny = min(self.convex_hulls[ind][:, 0, 1])

            maxx = maxx + handicap if maxx + handicap < len(self.image) else maxx
            maxy = maxy + handicap if maxy + handicap < len(self.image) else maxy
            minx = minx - handicap if minx - handicap < len(self.image) else minx
            miny = miny - handicap if miny - handicap < len(self.image) else miny


            assigned.append(ind)
            # we search for contours that are within found boundaries
            within = []

            for i in range(len(self.contours)):
                # if point in contour of the figure is within game min and max coordinates, assign it to this game
                if i not in assigned:
                    for vert in self.contours[i]:
                        if (minx < vert[0, 0] < maxx and miny < vert[0, 1] < maxy):
                            within.append(i)
                            assigned.append(i)
                            break

            if len(within) > 0:
                self.__handle_game(minx, miny, maxx, maxy)

    def __preprocess(self):  # Pablo liczę na to że wiesz co tu się dzieje, bo to twoje w końcu
        kernel = np.ones((4, 4), np.uint8)
        self._procImage = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self._procImage = cv.bilateralFilter(self._procImage, 9, 5, 5)
        self._procImage = cv.Canny(self._procImage, 20, 50)
        self._procImage = cv.morphologyEx(self._procImage, cv.MORPH_CLOSE, kernel)
        ret, self._procImage = cv.threshold(self._procImage, 127, 255, cv.THRESH_BINARY)

    def __filter_figures(self):  # filtering the figures - small ones are deleted
        i = 0
        while i < len(self.figures):  # iterate over all figures
            if cv.contourArea(self.figures[i].convHull) < 100:  # check if convex hull of the figure is big enough
                self.figures.pop(i)  # delete figure if it's too small
            else:
                i += 1

    def __filter_contours(self):  # filtering the contours - small ones are deleted
        i = 0
        while i < len(self.contours):  # iterate over all figures
            if cv.contourArea(self.convex_hulls[i]) < 100:  # check if convex hull of the contour is big enough
                self.contours.pop(i)  # delete contour if it's too small
                self.convex_hulls.pop(i)
            else:
                i += 1

    def __find_lines(self):  # function used for finding lines - probably will be changed
        linesP = cv.HoughLinesP(self._procImage, 1, np.pi / 240, 50, None, 60, 15)
        lines = []
        if len(linesP != 0):
            for i in range(len(linesP)):
                l = linesP[i]
                lines.append(Line([(l[0][0], l[0][1]), (l[0][2], l[0][3])]))
        return lines

    def create_figures(self):  # create figures out of the found contours
        self.figures = []  # array for storing created figures
        for i in range(len(self.contours)):  # iterate over all found contours
            center = (sum(self.contours[i][:, 0, 0]) / len(self.contours[i]),
                      sum(self.contours[i][:, 0, 1]) / len(self.contours[i]))

            # compute center of the figure - mean of all points' x and y coordinates
            tmpRad = []  # array for storing distance to the center of every point in the contour

            hull = cv.convexHull(self.contours[i])  # compute convex hull of the contour

            for j in range(len(self.contours[i])):  # iterate over every point in the contour
                tmpRad.append(sqrt((self.contours[i][j, 0, 0] - center[0]) ** 2 + (
                        self.contours[i][j, 0, 1] - center[1]) ** 2))  # distance to the center
            tmpRad = np.array(tmpRad)  # append the radius to the array
            self.figures.append(Figure(self.contours[i], center, hull,
                                       Measures(tmpRad, np.std(tmpRad), mean(tmpRad))))
            # create new figure from computed data and add it to the array in figures field
        self.__filter_figures()  # filter created figures - delete the ones that are too small

    def classify_figures(self):  # classify figures as either circle or cross
        for i in range(len(self.figures)):  # iterate over every figure
            hull_area = cv.contourArea(self.figures[i].convHull)  # compute area of convex hull of the figure
            con_area = cv.contourArea(self.figures[i].contour)  # compute area of the contour of the figure

            if con_area / hull_area < 0.5:
                self.figures[i].type = 'krzyzyk'
            else:
                self.figures[i].type = 'kolko'
            # right now squares also get classified as circle

    def find_games(self):  # find games of tic tac toe in the processed image
        self.games = []  # initialise array in games field
        assigned = []  # array for storing figures that have already been assigned to one game found on the image
        lines = self.__find_lines()  # find lines in the image - work in progress :)

        while len(assigned) < len(self.figures):  # look for new games until all figures are assigned to game
            maxi = 0  # initialise variable for storing index of largest not assigned figure
            while maxi in assigned:  # find first not assigned figure
                maxi += 1
            for i in range(len(self.figures)):  # find not assigned largest figure classified as cross
                if (self.figures[i].measure.avg > self.figures[maxi].measure.avg and i not in assigned and
                        self.figures[i].type == 'krzyzyk'):
                    maxi = i
            assigned.append(maxi)  # add found figure to the ones that has been assigned to games

            # initialise variables for searching for figures within the game
            maxx = 0
            maxy = 0
            minx = self.figures[maxi].contour[0, 0, 0]
            miny = self.figures[maxi].contour[0, 0, 1]

            # find min and max values of x and y coordinates in game
            for vert in self.figures[maxi].contour:
                if vert[0, 1] > maxy:
                    maxy = vert[0, 1]
                if vert[0, 1] < miny:
                    miny = vert[0, 1]
                if vert[0, 0] > maxx:
                    maxx = vert[0, 0]
                if vert[0, 0] < minx:
                    minx = vert[0, 0]

            within = []  # array for storing indexes of figures assigned to the game
            lines_within = []  # array for storing lines within game
            for i in range(len(self.figures)):
                # if point in contour of the figure is within game min and max coordinates, assign it ot this game
                if (minx < self.figures[i].center[0] < maxx and miny < self.figures[i].center[1] < maxy and
                        i not in assigned):
                    within.append(i)
                    assigned.append(i)

            for i in range(len(lines)):
                # if points constructing the line are within game min and max coordinates, add it to the game
                if ((minx < lines[i].points[0][0] < maxx and miny < lines[i].points[0][1] < maxy) and
                        (minx < lines[i].points[0][0] < maxx and miny < lines[i].points[0][1] < maxy)):
                    lines_within.append(lines[i])

            if len(within) != 0:  # if no figures were found within max figure, we don't create new game
                self.games.append(Game([(maxx, maxy), (minx, miny)], within, lines_within))

    def classify_games(self):
        for game in self.games:

            len_x = abs(game.border[0][0] - game.border[1][0]) // 3  # długość x jednego pola
            len_y = abs(game.border[0][1] - game.border[1][1]) // 3  # długość y jednego pola

            for index in range(9):  # iterate over 9 fields of tic-tac-toe game - in progress
                lx = game.border[1][0] + len_x * (index % 3)  # lewa granica pola
                rx = game.border[1][0] + len_x * ((index % 3) + 1)  # prawa granica pola
                dy = game.border[1][1] + len_y * (index // 3)  # górna granica pola
                uy = game.border[1][1] + len_y * ((index // 3) + 1)  # dolna granica pola

                in_field = []  # array for storing figures within

                for j in game.figures:  # iterate over all figures in the game
                    for vert in self.figures[j].contour:  # iterate over each point in figure's contour
                        if lx < vert[0, 0] < rx and dy < vert[0, 1] < uy:  # check if it is within the field
                            in_field.append(j)
                            break

                if len(in_field) == 0:  # if no figures are found in the field, we mark it as blank
                    self.image = cv.putText(self.image, '-', (lx + int(0.5 * len_x), uy - int(0.5 * len_y)),
                                            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                else:

                    maxi = in_field[0]  # initialise variable for storing index of the largest figure in the field

                    for index in in_field:  # iterate over all figures in the field
                        #  search for the largest figure
                        if self.figures[index].measure.avg > self.figures[maxi].measure.avg:
                            maxi = index

                    # check type of the largest figure and mark the field accordingly
                    if self.figures[maxi].type == 'kolko':
                        self.image = cv.putText(self.image, 'O', (lx + int(0.5 * len_x), uy - int(0.5 * len_y)),
                                                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                    else:
                        self.image = cv.putText(self.image, 'X', (lx + int(0.5 * len_x), uy - int(0.5 * len_y)),
                                                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)


class Measures:  # class for storing parameters of the figure
    def __init__(self, r: ndarray, st: float, av: float):
        self.radius = r  # array storing distance to the center of every point in the contour
        self.avg = av  # field containing average distance to the center in the figure
        self.stdev = st  # field containing standard deviation


class Figure:  # class storing data about the figure
    def __init__(self, con: ndarray, cen: tuple, hull: list, meas: Measures):
        self.contour = con  # field containing contour of the figure
        self.center = cen  # field containing center of the figure
        self.convHull = hull  # field containing convex hull of the figure
        self.measure = meas  # measures of the figure
        self.type = None  # field containing figure's type


class Line:  # class for storing line's coordinates
    def __init__(self, points: list):
        self.points = points  # points describing line
        #  compute parameters from line equation: y = ax + b
        try:
            self.dirx = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])  # y1-y2/x1-x2 - a in equation
            self.addx = points[0][1] - self.dirx * points[0][0]  # b in equation
            self.diry = None
            self.addy = None
        except ZeroDivisionError:  # if line is ideally vertical, compute parameters from equation x = ay + b
            self.dirx = None
            self.addx = None
            self.diry = (points[0][0] - points[1][0]) / (points[0][1] - points[1][1])  # a in equation
            self.addy = points[0][0] - self.diry * points[0][1]  # b in eqution
        self.len = sqrt((points[0][1] - points[1][1]) ** 2 + (points[0][0] - points[1][0]) ** 2)  # length of the line


class Game:  # class for storing game data and objects
    def __init__(self, border: list, fig: list, l: list):
        self.border = border  # 2 element array - 0 is point with max coordinates, 1 is point with min coordinates
        self.figures = fig  # array of indexes of figures within the game
        self.lines = l  # array of lines within the game
        self.__simplify_lines()  # try to connect smaller lines into larger ones - in progress

    def __simplify_lines(self):  # method used to connect smaller lines into larger ones
        length = np.array([])  # array for storing each line's length

        indexes = list(range(len(self.lines)))  # array storing indexes of the lines

        for i in indexes:  # iterate over all indexes
            j = i + 1  # get element after i-th element
            while j < len(indexes):  # iterate over all elements with indexes larger than i
                pi = self.lines[i]  # i-th line
                pj = self.lines[j]  # j-th line
                to_del = -1  # index of line to delete

                diffa = 5
                # parameter controlling how big the difference between 'a' parameters in line equations can be
                # in order to connect them
                diffb = 40
                # parameter controlling how big the difference between 'b' parameters in line equations can be
                # in order to connect them

                if (j != i and ((pi.dirx is not None and pj.dirx is not None and abs(pi.dirx - pj.dirx) < diffa
                                 and abs(pi.addx - pj.addx) < diffb))):
                    # when two lines are determined to be similar enough, we begin connecting them
                    ind = [(0, 0), (1, 0), (1, 1), (0, 1)]
                    dist = [sqrt((pi.points[i][0] - pj.points[j][0]) ** 2 + (pi.points[i][1] - pj.points[j][1]) ** 2)
                            for i, j in ind]
                    # check lengths of all possible combinations of merging the lines
                    p1, p2 = ind[dist.index(max(dist))]  # find the longest line

                    self.lines[i] = Line([pi.points[p1], pj.points[p2]])  # change i-th line for newly created one
                    to_del = j  # mark j-th line for deletion

                if to_del != -1:  # if line is marked for deletion, delete it from lines field, and delete the last
                    # element of indexes array
                    self.lines.pop(to_del)
                    indexes.pop()
                else:  # if line wasn't merged, continue with the next line
                    j += 1

            length = np.append(length, self.lines[i].len)  # add length of i-th line after all merges to the array
        avg = np.mean(length)  # compute average length
        i = 0
        while i < len(self.lines):  # delete lines with length under the average in game
            if self.lines[i].len < avg:
                self.lines.pop(i)
            else:
                i += 1


files = ["21", "23", "25", "26", "27", "29", "30", "32", "33", "34", "35", "36", "37"]
#files = ['3', '4']
for f in files:
    test = Loader("resources/SAM_06" + f + ".JPG")
    #test.create_figures()
    test.find_game()
    #test.classify_figures()
    #test.find_games()
    #test.classify_games()
'''
    im = test.image
    for i in range(len(test.games)):
        im2 = cv.rectangle(im, test.games[i].border[0], test.games[i].border[1], (0, 255, 0), 2)

    cv.imwrite('complete/' + f + '.png', im2)

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
    # ax.plot(test.figures[i].convHull[:, 0, 0], test.figures[i].convHull[:, 0, 1], linewidth=1, color='m')
    ax.plot(test.figures[i].center[0], test.figures[i].center[1], 'r+')

im = test.image

cv.imshow('window', im)

cv.waitKey(0)

plt.savefig("test1.png")'''
