class Point():
	def __init__(self, x, y, invert_y=None):
		self.x = x
		self.y = y

		if invert_y:
			self.y = invert_y - self.y

	def __str__(self):
		return "({}, {})".format(self.x, self.y)

class Line():
	def __init__(self, p1, p2):
		self.x0 = p1.x
		self.y0 = p1.y

		self.m = (p2.y - p1.y) / (p2.x - p1.x)

	def y(self, x):
		return round(self.m * (x - self.x0) + self.y0)

	def __str__(self):
		return "{} * (x - {}) + {}".format(self.m, self.x0, self.y0)

class LinearMap:
	def __init__(self, cp_index, points, scale = None, cp_range=256):
		self.cp_index = cp_index
		self.cp_range = cp_range
		self.points = []

		if points[0].x != 0:
			self.points.append(Point(0, 0))

		self.points.extend(points)

		if scale is not None:
			max_width = scale["w"]
			max_height = scale["h"]

			for point in self.points:
				point.x = round(cp_range / max_width * point.x)
				point.y = round(cp_range / max_height * point.y)

		if points[-1].x != self.cp_range - 1:
			self.points.append(Point(self.cp_range - 1, self.cp_range - 1))

		self.map_values = [0] * self.cp_range

		for i in range(1, len(self.points)):
			p1 = self.points[i - 1]
			p2 = self.points[i]
			line = Line(p1, p2)

			for x in range(p1.x, p2.x + 1):
				self.map_values[x] = line.y(x)

	def map(self, src, dst=None):
		if dst is None:
			dst = src

		for i in range(src.shape[0]):
			for j in range(src.shape[1]):
				dst[i][j][self.cp_index] = self.map_values[dst[i][j][self.cp_index]]

		return dst
