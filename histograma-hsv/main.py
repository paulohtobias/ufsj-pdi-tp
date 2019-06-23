import os
from http import server
import json
import cv2
import numpy as np
import sys
import tkinter
import PIL.Image, PIL.ImageTk

class Point():
	def __init__(self, x, y):
		self.x = x
		self.y = Point.ch - y

	@staticmethod
	def set_canvas_size(size):
		Point.cw = size["w"]
		Point.ch = size["h"]

class Line():
	def __init__(self, p1, p2):
		self.x0 = p1.x
		self.y0 = p1.y

		self.m = (p2.y - p1.y) / (p2.x - p1.x)

	def y(self, x):
		return round(self.m * (x - self.x0) + self.y0)

class JSONRequset(server.BaseHTTPRequestHandler):
	def _set_response(self, ct='text/json'):
		self.send_response(200)
		if ct is not None:
			self.send_header('Content-type', ct)
		self.end_headers()

	def do_GET(self):
		#content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
		self._set_response(ct=None)
		path = self.path if self.path != '/' else "/index.html"
		print(path)
		path = path[1:]
		self.wfile.write(open(path, "rb").read())

	def do_POST(self):
		content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
		post_data = self.rfile.read(content_length)  # <--- Gets the data itself

		data = json.loads(post_data.decode('utf-8'))
		print(json.dumps(data, indent=4))

		component = data["id"]
		Point.set_canvas_size(data["size"])

		points = [Point(0, data["size"]["h"])]
		points.extend([Point(p["x"], p["y"]) for p in data["points"]])
		points.append(Point(data["size"]["w"], 0))

		new_hist_map = [0] * 256
		for i in range(1, len(points)):
			p1 = points[i - 1]
			p2 = points[i]
			line = Line(p1, p2)

			for x in range(p1.x, p2.x):
				new_hist_map[x] = line.y(x)

		new_hist_map[255] = 255

		global img_hsv
		global new_img_hsv
		global new_img_rgb

		for i in range(new_img_hsv.shape[0]):
			for j in range(new_img_hsv.shape[1]):
				new_img_hsv[i][j][cmp_map[component]] = new_hist_map[img_hsv[i][j][cmp_map[component]]]

		new_img_rgb = cv2.cvtColor(new_img_hsv, cv2.COLOR_HSV2RGB)

		update_img()
		#cv2.imshow(component, new_img_rgb)
		#cv2.waitKey(1)
		#plt.imshow(new_img)
		#plt.show()

		self._set_response()
		self.wfile.write(b"{}")

def run(server_class=server.HTTPServer, handler_class=JSONRequset):
	server_address = ('', 4242)
	httpd = server_class(server_address, handler_class)
	httpd.serve_forever()

def resizePercent(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

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

def getHis(img):
	width, height, ch = img.shape
	histogram = [[0] * 256] * ch

	for j in range(width):
		for i in range(height):
			for k in range(ch):
				value = img[j][i][k]
				histogram[k][value] += 1

	return histogram

def update_img():
	global window
	global new_img_rgb
	global tk_canvas

	tk_canvas.delete("all")

	tk_img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(new_img_rgb))

	tk_canvas.create_image(0, 0, image=tk_img, anchor=tkinter.NW)

	window.update()

cmp_map = {
	"canvas-h": 0,
	"canvas-s": 1,
	"canvas-v": 2
}

try:
	image_name = sys.argv[1]
except IndexError:
	image_name = "/media/paulo/Arquivos/Paulo/Workspace/python/ufsj-pdi-tp/Imagens/1_85.jpg"
img_bgr = cv2.imread(image_name)
img_bgr = resizePercent(img_bgr, 60)
img_bgr = averageFilter(img_bgr, (5, 5))
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

new_img_hsv = img_hsv.copy()
new_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


with open("index.html", "w") as f:
	template = open("index-template.html").read()
	f.write(template)

# Create a window
window = tkinter.Tk()

tk_canvas = tkinter.Canvas(window, width=new_img_rgb.shape[1], height=new_img_rgb.shape[0])
tk_canvas.pack()

update_img()

print("Running server on http://localhost:4242")
run()
