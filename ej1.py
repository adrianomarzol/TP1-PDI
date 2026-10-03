import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import *

def EcualizacionHistogramaLocal(Imagen: np.ndarray, Ventana: Tuple[int,int]) -> np.ndarray:
    """
    Realiza un ecualizado de histograma local a la imagen. Tambien aplica medianBlur para el
    ruido.

    Recibe:
    Imagen: array de la imagen.
    Ventana: tupla con el alto y ancho de la ventana.

    Devuelve:
    CopiaImagen: array de la imagen con el ecualizado reailizado.
    """
    CopiaImagen = Imagen.copy()

    AltoImagen, AnchoImagen = CopiaImagen.shape
    AltoVentana, AnchoVentana = Ventana

    BordeSupInf = AltoVentana // 2
    BordeDerIzq = AnchoVentana // 2

    CopiaImagen = cv2.copyMakeBorder(CopiaImagen, BordeSupInf, BordeSupInf, BordeDerIzq, BordeDerIzq, cv2.BORDER_REPLICATE)

    for I in range(0, AltoImagen, AltoVentana):
        for J in range(0, AnchoImagen, AnchoVentana):
            Roi = CopiaImagen[I:I+AltoVentana,J:J+AnchoVentana]

            CopiaImagen[I:I+AltoVentana, J:J+AnchoVentana] = cv2.equalizeHist(Roi)

    CopiaImagen = cv2.medianBlur(CopiaImagen, 3)
    
    return CopiaImagen

ImagenConDetallesEscondidos = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

ImagenEcualizadoGlobal = cv2.equalizeHist(ImagenConDetallesEscondidos)

plt.imshow(ImagenEcualizadoGlobal, cmap = 'gray')
plt.show()

TamanosVentanas = [2, 16, 40, 100]

fig, axes = plt.subplots(1, 4, figsize = (18, 5))
for Indice, Ventana in enumerate(TamanosVentanas):
    ImagenEcualizadoLocal = EcualizacionHistogramaLocal(ImagenConDetallesEscondidos, (Ventana, Ventana))

    axes[Indice].imshow(ImagenEcualizadoLocal, cmap = 'gray')

plt.show()

ImagenEcualizadoLocal = EcualizacionHistogramaLocal(ImagenConDetallesEscondidos, (16,16))

plt.imshow(ImagenEcualizadoLocal, cmap = "gray")
plt.show()

print("""Objetos y detalles ocultos identificados:
    Al aplicar la ecualización local del histograma Imagen_con_objetos_ocultos.tiff,
    se divisan, dentro de las 5 regiones oscuras, objetos cuyos niveles de gris eran 
    indistinguibles del fondo. En la esquina superior derecha se distingue una línea 
    oblicua hacia la derecha de manera ascendente. En la esquina inferior derecha 
    vemos un circulo. En la esquina inferior izquierda se visualizan 4 líneas horizontales
    como si fueran renglones. En la esquina superior izquierda figura un cuadrado y
    por último, en el centro una letra ¨a¨ mínuscula """)
print(""" Análisis de la influencia del tamaño de las ventanas 
    Ventanas reducidas (ej: 2x2) :
    Al abarcar muy pocos píxeles, el histograma local se distribuye sobre un rango 
    estrecho, amplificando el ruido de alta frecuencia y produciendo un marcado 
    efecto de cuadrícula sin definir adecuadamente las formas.
    Ventanas de tamaño intermedio (ej: 40x40):
    Permiten capturar la variación local suficiente para realzar los objetos ocultos
    manteniendo su morfología y preservando un contraste suave y continuo respecto 
    de su entorno.
    Ventanas de gran tamaño (ej: 100x100):
    Cuanto más crece la ventana el histograma local tiende a parecerse más
    al histograma global de la imagen total. A su vez, se pierde la capacidad de
    discriminar contrastes tenues en regiones oscuras. El resultado converge al obtenido
    por cv2.equalizeHist() global. """)