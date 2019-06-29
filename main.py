# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
import linear_map
import components as cp
import moeda

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


#atribui um label para cada componente conectado
def getComponents(img_mask, img_color):
    rows, cols = img_mask.shape
    components = []

    for i in range(rows):
        for j in range(cols):
            if img_mask[i][j] == -255:
                component = cp.Component(img_color.shape)
                components.append(component)

                img_mask[i][j] = component.label
                linked = [(i, j)]
                component.add_pixel(i, j, img_color[i][j])
                while len(linked) > 0:
                    u, v = linked.pop()
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if (u + k) >= 0 and (u + k) < rows and (v + l) >= 0 and (v + l) < cols and img_mask[u + k][v + l] == -255:
                                img_mask[u + k][v + l] = component.label
                                linked.append((u + k, v + l))
                                component.add_pixel(u+k, v+l, img_color[u + k][v + l])


                component.crop()

    return components

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
    mask = [[1.0 / (mask_w * mask_h)] * mask_w] * mask_h

    avg_img = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for x in range(-mask_rh, mask_rh + 1):
                for y in range(-mask_rw, mask_rw + 1):
                    ix = index(i, x, img.shape[0])
                    jy = index(j, y, img.shape[1])

                    avg_img[i][j] += (mask[x + mask_rh][y + mask_rw] * img[ix][jy])

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

def imshow(img, title, cvt=cv2.COLOR_HSV2BGR):
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

    H, S, V = cv2.split(hsv)
    image = synthetic_removeBackground(hsv)
    imshow(image, 'filtro-range', None)

    # Erosão e Dilatação
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # kernel para erode dilate
    image = cv2.erode(image, kernel, iterations = 5)
    image = cv2.dilate(image, kernel, iterations = 5)
    imshow(image, 'erodil', None)

    components = getComponents(image * -1, hsv)

    for component in components:
        moedaIsolada = component.pixels
        imshow(moedaIsolada, 'aperte espaço')
        h, s, v = cv2.split(moedaIsolada)
        printHis(getHis(v))
        #cv2.waitKey(0)

    cv2.destroyAllWindows()
