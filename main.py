# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
import linear_map
import components as cp
import moeda as md

def printHis(histogram):
    plt.plot(histogram, color="black")
    plt.xticks([]), plt.yticks([])
    plt.show()

    return 0

#retorna o histograma de um determinado canal da imagem
def getHis(gray_img):
    width, height = gray_img.shape
    histogram = [0] * 256

    for j in range(0, width):
        for i in range(0, height):
            value = gray_img[j][i]
            histogram[value] += 1
    histogram[0] = 0
    return histogram

#retorna o pico do histograma
def maxHis(histogram):
    indMax = 0
    for i in range(1, 256):
        if histogram[i] > histogram[indMax]:
            indMax = i

    return indMax

#redimensiona a imagem
def resizePercent(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# redimensiona em tamanho absoluto.
# axis = 0: value é a altura.
# axis = 1: value é largura.
def resizeAbsolute(img, value, axis=1):
    width = img.shape[1]
    height = img.shape[0]

    ratio = float(value) / img.shape[axis]

    width = int(width * ratio)
    height = int(height * ratio)

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def averageFilter(img, mask_shape=(3, 3)):
    def index(axis, offset, limit):
        nv = axis + offset
        if nv < 0:
            return 0
        elif nv >= limit:
            return limit - 1
        else:
            return nv

    mask_w, mask_h = mask_shape
    mask_rw = mask_w // 2
    mask_rh = mask_h // 2

    avg_img = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cont = np.zeros(3)
            sum = np.zeros(3)
            for x in range(-mask_rh, mask_rh + 1):
                for y in range(-mask_rw, mask_rw + 1):
                    ix = index(i, x, img.shape[0])
                    jy = index(j, y, img.shape[1])

                    for channel in range(0, 3):
                        if img[ix][jy][channel] != 0:
                            sum[channel] += img[ix][jy][channel]
                            cont[channel] += 1

            for channel in range(0, 3):
                if cont[channel] != 0:
                    avg_img[i][j][channel] = sum[channel]/cont[channel]
                else:
                    avg_img[i][j][channel] = 0

    avg_img = np.array(avg_img, dtype=np.uint8)

    return avg_img

# Baseado em: https://stackoverflow.com/questions/4150171/how-to-create-a-density-plot-in-matplotlib
def getDensityFunction(gray_img):
    width, height = gray_img.shape
    data = []

    for j in range(0, width):
        for i in range(0, height):
            data += [gray_img[j][i]]

    density = gaussian_kde(data)
    xs = []
    for i in range(0, 256):
        xs += [i]

    density.covariance_factor = lambda : 1.0
    density._compute_covariance()
    return density(xs)

def cutPoints(densityFunction):
    pico = maxHis(densityFunction)
    min, max = 0, 255

    lastTan = 0
    for i in range(pico, 255):
        tan = (densityFunction[i] - densityFunction[i+1])
        if tan <= lastTan:
            max = i
            break
        lastTan = tan

    lastTan = 0
    for i in range(pico, 0, -1):
        tan = (densityFunction[i] - densityFunction[i-1])
        if tan <= lastTan:
            min = i
            break
        lastTan = tan

    return min, max

def imshow(img, title, cvt=cv2.COLOR_HSV2BGR, skip=True):
    if skip:
        return
    if cvt is not None:
        cv2.imshow(title, cv2.cvtColor(img, cvt))
    else:
        cv2.imshow(title, img)
    cv2.waitKey(0)

def natural_removeBackground(img):
    points_s = [linear_map.Point(48, 2)]
    points_v = [linear_map.Point(47, 217)]
    hsv = linear_map.LinearMap(1, points_s).map(img)
    hsv = linear_map.LinearMap(2, points_v).map(hsv)

    H, S, V = cv2.split(hsv)
    minH, maxH = (0, 255)#cutPoints(getDensityFunction(H))
    minS, maxS = cutPoints(getDensityFunction(S))
    minV, maxV = cutPoints(getDensityFunction(V))

    return ~cv2.inRange(hsv, np.array([minH, minS, minV]), np.array([maxH, maxS, maxV]))             # filtra o fundo da imagem

def synthetic_removeBackground(img):
    h = img[0][0][0]
    s = img[0][0][1]
    v = img[0][0][2]

    return ~cv2.inRange(img, np.array([h-1, s-1, v-1]), np.array([h+1, s+1, v+1]))

if __name__ == "__main__":
    # Abre e redimensiona a imagem.
    bgr_img = cv2.imread(sys.argv[1])
    bgr_img = resizeAbsolute(bgr_img, 360)
    #imshow(bgr_img, "original", None)

    #bgr_img = averageFilter(bgr_img, (3,3))
    imshow(bgr_img, 'média', None)

    # Conversão pra HSV.
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    image = synthetic_removeBackground(hsv)
    imshow(image, 'filtro-range', None)

    # Erosão e Dilatação
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # kernel para erode dilate
    image = cv2.erode(image, kernel, iterations = 5)
    image = cv2.dilate(image, kernel, iterations = 4)
    imshow(image, 'erodil', None)

    components = cp.getComponents(image * -1, hsv)


    for component in components:
        print(component)

    cv2.destroyAllWindows()
