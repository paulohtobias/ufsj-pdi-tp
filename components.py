# -*- coding: utf-8 -*-

import cv2
import numpy as np
import moeda as md
from main import averageFilter

class Component:
	_label = 0
	def __init__(self, img_shape):
		self.label = Component._label
		Component._label += 1

		self.pixels = np.zeros(img_shape, dtype=np.uint8)
		self.area = 0

		self.prata = 0.0
		self.bronze = 0.0

		self.x_left = img_shape[1]
		self.x_right = 0
		self.y_top = img_shape[0]
		self.y_bottom = 0

	def add_pixel(self, i, j, value):
		self.pixels[i][j] = value.copy()
		self.area += 1

		# Atualizando as dimensões do componente.
		if j < self.x_left:
			self.x_left = j
		if j > self.x_right:
			self.x_right = j
		if i < self.y_top:
			self.y_top = i
		if i > self.y_bottom:
			self.y_bottom = i

	# https://stackoverflow.com/a/15589825
	def crop(self):
		self.pixels = averageFilter(self.pixels[self.y_top:self.y_bottom, self.x_left:self.x_right], (7, 7))

		prata = cv2.inRange(self.pixels, md.prata.hsv.min, md.prata.hsv.max)
		bronze = cv2.inRange(self.pixels, md.bronze.hsv.min, md.bronze.hsv.max)

		self.prata = Component.percent(prata)
		self.bronze = Component.percent(bronze)

	@staticmethod
	def percent(range_img):
		total = 0.0
		for i in range(range_img.shape[0]):
			for j in range(range_img.shape[1]):
				if range_img[i][j] != 0:
					total += 1.0

		return total / float(range_img.shape[0] * range_img.shape[1])

	def __str__(self):
		return "Componente {} ({} pixels) - ({}% prata, {}% bronze)".format(self.label, self.area, self.prata * 100, self.bronze * 100)

# Retorna uma lista de Componentes de uma imagem
def getComponents(img_mask, img_color):
    rows, cols = img_mask.shape
    components = []

    for i in range(rows):
        for j in range(cols):
            if img_mask[i][j] == -255:
                component = Component(img_color.shape)
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
