# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
import linear_map

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

    ratio = value / img.shape[axis]

    width = int(width * ratio)
    height = int(height * ratio)

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


#atribui um label para cada componente conectado
def bwLabel(img):
    rows, cols = img.shape
    label = 0

    for i in range(rows):
        for j in range(cols):
            if img[i][j] == -255:
                label += 1
                img[i][j] = label
                linked = [(i, j)]
                while len(linked) > 0:
                    u, v = linked.pop()
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if (u + k) >= 0 and (u + k) < rows and (v + l) >= 0 and (v + l) < cols and img[u + k][v + l] == -255:
                                img[u + k][v + l] = label
                                linked.append((u + k, v + l))

    return label, img

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
    mask = [[1 / (mask_w * mask_h)] * mask_w] * mask_h

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

    print(min, max)
    return min, max

def cutPoints2(densityFunction):
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

    print(min, max)
    return min, max

def imshow(img, title, cvt=cv2.COLOR_HSV2BGR):
    if cvt is not None:
        cv2.imshow(title, cv2.cvtColor(img, cvt))
    else:
        cv2.imshow(title, img)
    cv2.waitKey(0)

# Abre e redimensiona a imagem.
bgr_img = cv2.imread(sys.argv[1])
bgr_img = resizeAbsolute(bgr_img, 360)
imshow(bgr_img, "original", None)

# Filtro de média.
#bgr_img = averageFilter(bgr_img, (5,5))
bgr_img = cv2.blur(bgr_img, (7,7))
imshow(bgr_img, 'média', None)

# Conversão pra HSV.
hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

# Mapeamento de componentes
points_s = [linear_map.Point(48, 2)]
points_v = [linear_map.Point(47, 217)]
hsv = linear_map.LinearMap(1, points_s).map(hsv)
hsv = linear_map.LinearMap(2, points_v).map(hsv)
imshow(hsv, 'eq-sv')

# Filtro de componentes
H, S, V = cv2.split(hsv)
minH, maxH = (0, 255)#cutPoints(getDensityFunction(H))
minS, maxS = cutPoints(getDensityFunction(S))
minV, maxV = cutPoints(getDensityFunction(V))

image = cv2.inRange(hsv, np.array([minH, minS, minV]), np.array([maxH, maxS, maxV]))             # filtra o fundo da imagem
image = ~image                                                # inverte as cores da imagem

imshow(image, 'filtro-range', None)

# Erosão e Dilatação
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))  # kernel para erode dilate
image = cv2.erode(image, kernel, iterations = 1)
image = cv2.dilate(image, kernel, iterations = 1)
imshow(image, 'erodil', None)

qtddMoedas, image = bwLabel(image*(-1))

for i in range(1, qtddMoedas+1):
    moedaIsolada = np.uint8(image == i)*255
    imgTmp = cv2.merge((moedaIsolada & H, moedaIsolada & S, moedaIsolada & V))
    imgTmp = cv2.cvtColor(imgTmp, cv2.COLOR_HSV2BGR) # volta para bgr pra poder exibir
    cv2.imshow('aperte espaço', imgTmp)
    cv2.waitKey(0)

cv2.destroyAllWindows()
