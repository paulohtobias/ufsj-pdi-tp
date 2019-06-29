#!/bin/bash

# "./test.sh" executa o programa para todas as imagens.
# "./test.sh Imagens/nome_imagem.jpg" executa o prgrama para uma imagem específica.

IMGS="$(ls Imagens | egrep *.jpg)"

run()
{
	echo $3
	$@
	echo ""
}

if [ $1 ]; then
	if [ "$1" = "--help" ]; then
		echo "\"$0\"                         - executa o programa para todas as imagens."
		echo "\"$0 Imagens/nome_imagem.jpg\" - executa o programa para uma imagem específica."
	else
		run python3 main.py $1
	fi
	exit
fi

for IMG in $IMGS; do
	run python3 main.py Imagens/$IMG
	#python3 main.py Imagens/$IMG
done
