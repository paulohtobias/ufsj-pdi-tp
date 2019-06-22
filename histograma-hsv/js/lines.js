// Utils
function getMousePos(canvas, ev) {
	let rect = canvas.getBoundingClientRect();
	return {
		x: Math.round(ev.clientX - rect.left),
		y: Math.round(ev.clientY - rect.top)
	};
}


class HistCanvas {
	// static
	static get rs() {
		return 3;
	}

	constructor(id) {
		this.id = id;
		this.canvas = document.getElementById(id);
		this.ctx = this.canvas.getContext('2d');
		this.width = this.canvas.width;
		this.height = this.canvas.height;
		this.points = [];

		this.mouse = {
			drag: -1,
			dz: 0
		};
		this.canvas.addEventListener("mousedown", this.add_point.bind(this));
		this.canvas.addEventListener("mousemove", this.drag_point.bind(this));
		this.canvas.addEventListener("mouseup", this.update_image.bind(this));
	}

	add_point(ev) {
		let mp = getMousePos(this.canvas, ev);
		console.log(mp.x, mp.y);

		// Inserindo o ponto mantendo a ordenação.
		let low = 0,
			high = this.points.length;
		while (low < high) {
			let mid = (low + high) >>> 1;
			if (this.points[mid].x < mp.x) low = mid + 1;
			else high = mid;
		}
		this.mouse.drag = low;

		let point_in_line = this.point_in_line(mp);

		if (point_in_line == -1) {
			this.points.splice(low, 0, mp);
			console.log(this.points);
		} else {
			this.mouse.drag = point_in_line;
		}

		this.draw();
	}

	drag_point(ev) {
		if (this.mouse.drag >= 0) {
			let mp = getMousePos(this.canvas, ev);

			this.points[this.mouse.drag] = mp;

			this.draw();
		}
	}

	update_image() {
		this.mouse.drag = -1;

		$.ajax({
			url: "http://localhost:4242",
			method: "POST",
			data: JSON.stringify({
				id: this.id,
				size: {
					w: this.width,
					h: this.height
				},
				points: this.points
			})
		});
	}

	draw() {
		this.ctx.clearRect(0, 0, this.width, this.height);

		// Linha
		this.ctx.beginPath();
		this.ctx.moveTo(0, this.height);
		this.points.forEach(point => {
			this.ctx.lineTo(point.x, point.y);
		});
		this.ctx.lineTo(this.width, 0);
		this.ctx.stroke();

		// Pontos
		this.ctx.beginPath();
		this.points.forEach(point => {
			this.ctx.rect(point.x - HistCanvas.rs, point.y - HistCanvas.rs, HistCanvas.rs * 2, HistCanvas.rs * 2);
			this.ctx.fill();
		});
	}

	point_in_line(p, max = 0) {
		if (this.points.length == 0) {
			return -1;
		}

		for (let i = 0; i < this.points.length; i++) {
			let point = this.points[i];

			if (p.x >= point.x - HistCanvas.rs && p.x <= point.x + HistCanvas.rs * 2 &&
				p.y >= point.y - HistCanvas.rs && p.y <= point.y + HistCanvas.rs * 2)
			{
				return i;
			}

		}

		return -1;
	}
}

// get the canvas element using the DOM
var canvas = {
	h: new HistCanvas('canvas-h'),
	s: new HistCanvas('canvas-s'),
	v: new HistCanvas('canvas-v')
}

function draw_all() {
	canvas.h.draw();
	canvas.s.draw();
	canvas.v.draw();
}

function update_all() {
	canvas.h.update_image();
	canvas.s.update_image();
	canvas.v.update_image();
}

function draw_hist(id, hist) {
	let canvas = document.getElementById(id);
	let ctx = canvas.getContext('2d');
	let width = canvas.width,
		height = canvas.height;

	let max = Math.max(...hist);

	for (let i = 0; i < hist.length; i++) {
		hist[i] = height * hist[i] / max;
	}


	// Desenhando

	ctx.clearRect(0, 0, width, height);

	for (let i = 0; i < hist.length; i++) {
		ctx.beginPath();
		ctx.moveTo(i, height);
		ctx.lineTo(i, hist[i]);
		ctx.stroke();
	}
}
