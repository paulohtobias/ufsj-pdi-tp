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
	def __init__(self, valor):
		self.valor = valor

	def __str__(self):
		return "R$%.2f" % (self.valor)

class MoedaNFException(Exception):
	def __init__(self):
		super(MoedaNFException, self).__init__("Não foi possível identificar a moeda")

def obter_maior_valor(componente, qtd_menores):
	# Técnica avançada de IA.
	if componente.prata >= 0.2:
		if componente.bronze >= 0.2:
			return Moeda(1.0)
		else:
			return Moeda(0.50)
	elif componente.bronze >= 0.2:
		if qtd_menores >= 2:
			return Moeda(0.25)
		elif qtd_menores == 1:
			# Consulta proporção
			raise Exception("TODO: consulta proporção")

	raise MoedaNFException

__proporcao = {
	1.0: [
		{
			"area_r": 0.85,
			"valor": 0.25
		},
		{
			"area_r": 0.73,
			"valor": 0.50
		},
		{
			"area_r":  0.66,
			"valor": 0.05
		},
		{
			"area_r": 0.53,
			"valor": 0.10
		}
	],
	0.25: [
		{
			"area_r": 0.85,
			"valor": 0.50
		},
		{
			"area_r": 0.77,
			"valor": 0.05
		},
		{
			"area_r": 0.63,
			"valor": 0.10
		}
	],
	0.50: [
		{
			"area_r": 0.90,
			"valor": 0.05
		},
		{
			"area_r": 0.73,
			"valor": 0.10
		}
	],
	0.5: [
		{
			"area_r": 0.81,
			"valor": 0.10
		}
	]
}
def valor_por_proporcao(componente, maior_moeda):
	global __proporcao
	for prop in __proporcao[maior_moeda.valor]:
		if abs(componente.relative_area - prop["area_r"]) < 0.04:
			return Moeda(prop["valor"])
	raise MoedaNFException

prata = CorMoeda('hsv', [48, 30, 100], alcance=[30, 29, 50])
bronze = CorMoeda('hsv', [24, 135, 135], alcance=[15, 85, 85])
