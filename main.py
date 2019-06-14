import cv2
import numpy as np
import matplotlib.pyplot as plt

bgr_img = cv2.imread('Imagens/2_10.jpg')
kernel = np.ones((3,3),np.uint8)  # kernel para erode dilate
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

#redimenciona a imagem
def resizePercent(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

bgr_img = resizePercent(bgr_img, 60)
hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

maxHisH = maxHis(getHis(H))
maxHisS = maxHis(getHis(S))
maxHisV = maxHis(getHis(V))

lower = np.array([maxHisH - margin_h, maxHisS - margin_s, maxHisV - margin_v])
upper = np.array([maxHisH + margin_h, maxHisS + margin_s, maxHisV + margin_v])

image = cv2.inRange(hsv, lower, upper)                          # filtra o fundo da imagem
image = ~image                                                  # inverte as cores da imagem

image = cv2.erode(image,kernel,iterations = 1)
image = cv2.dilate(image,kernel,iterations = 1)

print maxHisH, maxHisS, maxHisV

cv2.imshow('original', bgr_img)
cv2.imshow('binaria', image)
cv2.waitKey(0)

cv2.destroyAllWindows()
















#
