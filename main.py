# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

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
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

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


bgr_img = cv2.imread(sys.argv[1])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))  # kernel para erode dilate

bgr_img = resizePercent(bgr_img, 60)
#cv2.imshow('original', bgr_img)
bgr_img = averageFilter(bgr_img, (5,5))
#cv2.imshow('média', bgr_img)
#bgr_img = cv2.blur(bgr_img, (3,3))
hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

#printHis(getHis(H))
#printHis(getHis(S))
#printHis(getHis(V))
#exit()
maxHisH = maxHis(getHis(H))
maxHisS = maxHis(getHis(S))
maxHisV = maxHis(getHis(V))

margin_h = 170
margin_s = 100
margin_v = 60

lower = np.array([maxHisH - margin_h, maxHisS - margin_s, maxHisV - margin_v])
upper = np.array([maxHisH + margin_h, maxHisS + margin_s, maxHisV + margin_v])

image = cv2.inRange(hsv, lower, upper)  # filtra o fundo da imagem
image = ~image                          # inverte as cores da imagem

image = cv2.erode(image, kernel, iterations = 1)
image = cv2.dilate(image, kernel, iterations = 1)

qtddMoedas, image = bwLabel(image*(-1))

for i in range(1, qtddMoedas+1):
    moedaIsolada = np.uint8(image == i)*255
    imgTmp = cv2.merge((moedaIsolada & H, moedaIsolada & S, moedaIsolada & V))
    cv2.imshow('aperte espaço', imgTmp)
    cv2.waitKey(0)

print(maxHisH, maxHisS, maxHisV)
cv2.destroyAllWindows()
