import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import *

def EcualizacionHistogramaLocal(Imagen: np.ndarray, Ventana: Tuple[int,int]) -> np.ndarray:

    """
    Realiza un ecualizado de histograma local a la imagen. La ecualización local es una técnica de procesamiendo de imágenes que redistribuye las intensidades de los 
    píxeles para mejorar su contraste y facilitar la visualización de sus detalles

    Recibe:
    Imagen: array de la imagen.
    Ventana: tupla con el alto y ancho de la ventana.

    Devuelve:
    CopiaImagen: array de la imagen con el ecualizado reailizado.
    """
    AltoImagen, AnchoImagen = Imagen.shape
    AltoVentana, AnchoVentana = Ventana

    BordeSupInf = AltoVentana // 2
    BordeDerIzq = AnchoVentana // 2

    ImagenExtendida = cv2.copyMakeBorder(Imagen, BordeSupInf, BordeSupInf, BordeDerIzq, BordeDerIzq, cv2.BORDER_REPLICATE)
    CopiaImagen = np.zeros_like(Imagen)

    for I in range(AltoImagen):
        for J in range(AnchoImagen):
            Roi = ImagenExtendida[I:I+AltoVentana, J:J+AnchoVentana]
            Histograma = np.bincount(Roi.ravel(), minlength=256)
            CDF = np.cumsum(Histograma) / Roi.size
            PixelCentral = ImagenExtendida[I+BordeSupInf, J+BordeDerIzq]
            CopiaImagen[I, J] = np.uint8(255 * CDF[PixelCentral])

    return CopiaImagen

ImagenConDetallesEscondidos = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

ImagenEcualizadoGlobal = cv2.equalizeHist(ImagenConDetallesEscondidos)

plt.imshow(ImagenEcualizadoGlobal, cmap = 'gray') 
plt.title("Ecualización global (cv2.equalizeHist)")
plt.show() 

TamanosVentanas = [3, 9, 25]

Resultados = []
for Ventana in TamanosVentanas:
    Resultado = EcualizacionHistogramaLocal(ImagenConDetallesEscondidos, (Ventana, Ventana))
    Resultados.append(Resultado)
    plt.figure()
    plt.imshow(Resultado, cmap='gray')
    plt.title(f"Ecualización local {Ventana} x {Ventana}")
    plt.show(block=False)

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.title("Imagen original")
plt.imshow(ImagenConDetallesEscondidos, cmap='gray')

for Indice, Ventana in enumerate(TamanosVentanas):
    plt.subplot(2, 2, Indice + 2)
    plt.title(f"Ecualización local - {Ventana}x{Ventana}")
    plt.imshow(Resultados[Indice], cmap='gray')
plt.subplots_adjust(hspace=0.4)
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
    Ventanas reducidas (ej: 3x3) :
    Al abarcar muy pocos píxeles, el histograma local se estima con muy pocas muestras,
    amplificando el ruido de alta frecuencia y sin definir adecuadamente las formas.
    Ventanas de tamaño intermedio (ej: 9x9):
    Permiten capturar la variación local suficiente para realzar los objetos ocultos
    manteniendo su morfología y preservando un contraste suave y continuo respecto 
    de su entorno.
    Ventanas de gran tamaño (ej: 25x25):
    Cuanto más crece la ventana el histograma local tiende a parecerse más
    al histograma global de la imagen total. A su vez, se pierde la capacidad de
    discriminar contrastes tenues en regiones oscuras. El resultado se aproxima cada vez
    más al obtenido por cv2.equalizeHist() global. """)