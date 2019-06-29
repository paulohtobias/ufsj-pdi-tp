import numpy as np

class Limiar():
	def __init__(self, min, max):
		self.min = np.array(min, dtype=np.uint8)
		self.max = np.array(max, dtype=np.uint8)

class CorMoeda:
	def __init__(self, espaco, valor, **kwargs):
		try:
			alcance = kwargs['alcance']
			minimo = [v - a for v, a in zip(valor, alcance)]
			maximo = [v + a for v, a in zip(valor, alcance)]
		except KeyError:
			minimo = [v - m for v, m in zip(valor, kwargs['min'])]
			maximo = [v + m for v, m in zip(valor, kwargs['max'])]
		self.__dict__[espaco] = Limiar(minimo, maximo)

class Moeda:
	def __init__(self, valor, cor):
		self.valor = valor
		self.cor = cor

moedas = [
	Moeda(50, CorMoeda('hsv', [48, 30, 100], alcance=[30, 29, 50])),
	Moeda(25, CorMoeda('hsv', [24, 135, 135], alcance=[15, 85, 85]))
]
