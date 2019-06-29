import cv2
import numpy as np

class Component:
	_label = 0
	def __init__(self, img_shape):
		self.label = Component._label
		Component._label += 1

		self.pixels = np.zeros(img_shape, dtype=np.uint8)
		self.area = 0

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
		self.pixels = self.pixels[self.y_top:self.y_bottom, self.x_left:self.x_right]
