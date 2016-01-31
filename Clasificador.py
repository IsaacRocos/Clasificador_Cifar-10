# ---------------------------------------------------------------------------
# isaac
# Clasificador Ciffar-10. 
# Python script to classify a single image 
# using cifar-10 caffe example model. 
# ---------------------------------------------------------------------------

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
import pyglet
import caffe
import sys
import os


PROTOTIPO = "CNN/cifar10.prototxt"
MODEL = "CNN/cifar10_60000.caffemodel.h5"
IMAGE_DIR  = "set_cifar10/imgs/" # using cifar-10 png images from Kagle to test
LABEL_FILE = "set_cifar10/lblCIFAR10.csv"
FILE_FORMAT = ".png"
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

CNN = caffe.Classifier(PROTOTIPO, MODEL)

csv = open(LABEL_FILE,"r")
etiquetas = (csv.read()).split('\n')
etiquetas.pop() #eliminar espacio en blanco


def classify(imgPath , contador):
	print "\nEJECUTANDO PRUEBA"
	print "\nClasificando imagen..."
	out = CNN.forward()
	imagenIN = caffe.io.load_image(imgPath, color=True)	
	imgCIFAR10 = img2CIFARFormat(imagenIN)
	#----------------------------Show images ---------------------
	#plt.imshow(imagenIN)
	#plt.show()
	#------------------------------------------------------------------
	clasificacion = CNN.predict([imgCIFAR10])	
	res = int(clasificacion[0].argmax())
	print "Classified: ", res, "::",  out['prob'][0][res], " Expected: " , etiquetas[contador]


def img2CIFARFormat(imagen):
	#Description cifar-10
	#The first entries contain the red channel values, the next the green, 
	#and the final the blue. 
	#The image is stored in row-major order, so that the first entries of 
	#the array are the red channel values of the first row of the image.
	rearranged = np.zeros_like(imagen)
	#RED   [0][0][0]   <-> [10][21][0] 
	rf = 0
	rc = 0
	rv = 0
	#GREEN  [10][20][1] <-> [21][10][1] 
	gf = 10
	gc = 21
	gv = 1
	#BLUE   [21][10][2] <-> [31][31][2]
	bf = 21
	bc = 10
	bv = 2

	for fila in range (32):
		for columna in range (32):
			if (rv >2):	
				rc +=1
				rv = 0
			if (gv >2):	
				gc +=1
				gv = 0
			if (bv >2):
				bc +=1
				bv = 0
			if (rc >31):	
				rf +=1
				rc = 0
			if (gc >31):	
				gf +=1
				gc = 0
			if (bc >31):
				bf +=1
				bc = 0
			'''
			if(fila > 30):
				print "ROJO " , "[",rf,"]", "[",rc,"]", "[",rv,"]"
				print "VERDE" , "[",gf,"]", "[",gc,"]", "[",gv,"]"
				print "AZUL " , "[",bf,"]", "[",bc,"]", "[",bv,"]"
				print ""
			'''
			rearranged[rf][rc][rv] = imagen[fila][columna][0]
			rearranged[gf][gc][gv] = imagen[fila][columna][1]
			rearranged[bf][bc][bv] = imagen[fila][columna][2]
			rv +=1
			gv +=1
			bv +=1
	return rearranged


def main():	
	for contador in range (100):
		classify(IMAGE_DIR + str(contador+1) + FILE_FORMAT,contador)



if __name_ == '__main__':main()
