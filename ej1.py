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