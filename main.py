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
        self.__filter_contours()  # delete small contours from the arra

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

    def __detect_lines(self, image, secImage, type: int):  # By Borubasz
        lines = cv.HoughLines(image, 1, np.pi / 180, 50, None, 0, 0)
        strong_lines = []
        maxy, maxx = image.shape[:2]
        avg = (maxy+maxx+20)/2
        if lines is None:
            return None
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
                if (j[0][0] - 0.15*avg <= i[0][0] <= j[0][0] + 0.15*avg) and (
                        j[0][1] - np.pi / 18 <= i[0][1] <= j[0][1] + np.pi / 18):
                    chk += 1
            if chk == 0 and len(strong_lines) < type:
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

    def __handle_game(self, minx: int, miny: int, maxx: int, maxy: int, type: int):  # By Borubasz
        game = self._procImage[miny:maxy, minx:maxx]  # cutting out founded field
        tmpGame = self.image[miny:maxy, minx:maxx]  # cutting out founded field
        '''
        print(minx, ' ', maxx, ' ', miny, ' ', maxy)
        cv.imshow('window', tmpGame)
        cv.waitKey(0)
        cv.destroyAllWindows()'''
        lines_eq = self.__detect_lines(game, tmpGame, type)
        if lines_eq is None or len(lines_eq) < type:
            return False

        if type == 4:
            points_of_interest = [[0, 0], [0, (maxx - minx)], [(maxy - miny), 0], [(maxy - miny), (maxx - minx)]]  # [y, x]
        else:
            points_of_interest = []

        for i in range(int(type/2)):
            for j in range(int(type/2), type):
                if lines_eq[i][0] == -inf:
                    x = lines_eq[i][2]
                    y = lines_eq[j][0]*x+lines_eq[j][1]
                else:
                    x = (lines_eq[i][1] - lines_eq[j][1])/(lines_eq[i][0] - lines_eq[j][0])
                    y = lines_eq[i][0]*x+lines_eq[i][1]
                    print(i, ' ', j, ' ', x, ' ', y)
                    points_of_interest.append([int(y), int(x)])
        
        print("Punkty: ",  len(points_of_interest))

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

                if con_area > 0.001*(maxx-minx)*(maxy-miny) and con_area / hull_area > 0.5:
                    tmpGame = cv.putText(tmpGame, 'O', center, cv.FONT_HERSHEY_SIMPLEX,
                                         1, (255, 0, 255), 3)
                elif con_area > 0.001*(maxx-minx)*(maxy-miny) and con_area / hull_area <= 0.5:
                    tmpGame = cv.putText(tmpGame, 'X', center, cv.FONT_HERSHEY_SIMPLEX,
                                         1, (255, 0, 255), 3)
        return True
        '''
        cv.imshow('window', tmpGame)
        cv.waitKey(0)
        cv.destroyAllWindows()
        '''

    def find_game(self):  # By Borubasz: this little function finds contour with biggest convex hull
        # if you would take a look at the photo with drawn convex hull you would find that convex hull pretty nicely
        # outlines game field
        assigned = []
        while len(assigned) < len(self.convex_hulls):
            ind = 0

            handicap = 10
            for i in range(0, len(self.convex_hulls)):
                if (i not in assigned and
                        cv.contourArea(self.convex_hulls[ind]) < cv.contourArea(self.convex_hulls[i])):
                    ind = i

            hull_area = cv.contourArea(self.convex_hulls[ind])  # compute area of convex hull of the figure
            con_area = cv.contourArea(self.contours[ind])  # compute area of the contour of the figure

            if con_area / hull_area < 0.5:
                type = 4
            else:
                type = 8

            # we take maxes and mins from hull to find rectangle including game
            maxx = max(self.convex_hulls[ind][:, 0, 0])
            maxy = max(self.convex_hulls[ind][:, 0, 1])
            minx = min(self.convex_hulls[ind][:, 0, 0])
            miny = min(self.convex_hulls[ind][:, 0, 1])

            maxx_im = maxx + handicap if maxx + handicap <= len(self.image) else len(self.image)
            maxy_im = maxy + handicap if maxy + handicap <= len(self.image) else len(self.image)
            minx_im = minx - handicap if minx - handicap >= 0 else 0
            miny_im = miny - handicap if miny - handicap >= 0 else 0

            assigned.append(ind)
            # we search for contours that are within found boundaries
            within = []

            for i in range(len(self.contours)):
                # if point in contour of the figure is within game min and max coordinates, assign it to this game
                if i not in assigned:
                    for vert in self.contours[i]:
                        if minx < vert[0, 0] < maxx and miny < vert[0, 1] < maxy:
                            within.append(i)
                            assigned.append(i)
                            break

            if len(within) > 0:
                b = True
                b = self.__handle_game(minx_im, miny_im, maxx_im, maxy_im, type)
                if b is True:
                    cv.rectangle(self.image, (minx, miny), (maxx, maxy), (125, 125, 0), 1)
        cv.imwrite("obsluzone-zdjecie.jpg", self.image)
        cv.imshow('window', self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def __preprocess(self):  # Pablo liczę na to że wiesz co tu się dzieje, bo to twoje w końcu
        kernel = np.ones((4, 4), np.uint8)
        self._procImage = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self._procImage = cv.bilateralFilter(self._procImage, 9, 5, 5)
        self._procImage = cv.Canny(self._procImage, 20, 50)
        self._procImage = cv.morphologyEx(self._procImage, cv.MORPH_CLOSE, kernel)
        ret, self._procImage = cv.threshold(self._procImage, 127, 255, cv.THRESH_BINARY)

    def __filter_contours(self):  # filtering the contours - small ones are deleted
        i = 0
        while i < len(self.contours):  # iterate over all figures
            if cv.contourArea(self.convex_hulls[i]) < 100:  # check if convex hull of the contour is big enough
                self.contours.pop(i)  # delete contour if it's too small
                self.convex_hulls.pop(i)
            else:
                i += 1


#files = ["21", "23", "25", "26", "27", "29", "30", "32", "33", "34", "35", "36", "37"]
#files = ['3', '4']
files = ['32']
for f in files:
    test = Loader("resources/Łatwe/SAM_06" + f + ".JPG")
    test.find_game()
