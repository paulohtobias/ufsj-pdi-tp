# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

bgr_img = cv2.imread(sys.argv[1])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))  # kernel para erode dilate
margin_h = 250
margin_s = 100
margin_v = 60

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
    max = histogram[0]
    indMax = 0
    for i in range(1, 256):
        if histogram[i] > max:
            max = histogram[i]
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

def averageFilter(img):
    mask = [[1*(1/9)]*3]*3
    b, g, r = cv2.split(img)
    b_aux = g_aux = r_aux = np.zeros((img.shape[0], img.shape[1]))

    for i in range(1, (img.shape[0]-1)):
        for j in range(1, (img.shape[1]-1)):
            for x in range(-len(mask)//2, len(mask)//2):
                for y in range(-len(mask)//2, len(mask)//2):
                    b_aux[i][j] += (mask[x+1][y+1] * b[i+x][j+y])
                    g_aux[i][j] += (mask[x+1][y+1] * g[i+x][j+y])
                    r_aux[i][j] += (mask[x+1][y+1] * r[i+x][j+y])

    img = cv2.merge([b_aux, g_aux, r_aux])
    img = np.array(img, dtype=np.uint8)
    
    return img


bgr_img = resizePercent(bgr_img, 60)
#bgr_img = averageFilter(bgr_img)
bgr_img = cv2.blur(bgr_img, (3,3))
hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

maxHisH = maxHis(getHis(H))
maxHisS = maxHis(getHis(S))
maxHisV = maxHis(getHis(V))

lower = np.array([maxHisH - margin_h, maxHisS - margin_s, maxHisV - margin_v])
upper = np.array([maxHisH + margin_h, maxHisS + margin_s, maxHisV + margin_v])

image = cv2.inRange(hsv, lower, upper)                          # filtra o fundo da imagem
image = ~image                                                # inverte as cores da imagem

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
