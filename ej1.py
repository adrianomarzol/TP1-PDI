import matplotlib.pyplot as plt
import numpy as np
import cv2

def EcualizacionHistogramaLocal(Imagen, Ventana):
    """
    """
    CopiaImagen = Imagen.copy()

    AltoImagen, AnchoImagen = CopiaImagen.shape
    AltoVentana, AnchoVentana = Ventana

    for I in range(0, AltoImagen, AltoVentana):
        for J in range(0, AnchoImagen, AnchoVentana):
            Roi = CopiaImagen[I:I+AltoVentana,J:J+AnchoVentana]

            CopiaImagen[I:I+AltoVentana, J:J+AnchoVentana] = cv2.equalizeHist(Roi)

    CopiaImagen = cv2.medianBlur(CopiaImagen, 3)
    
    return CopiaImagen

ImagenConDetallesEscondidos = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

ImagenEcualizadoLocal = EcualizacionHistogramaLocal(ImagenConDetallesEscondidos, (14,14))

plt.imshow(ImagenEcualizadoLocal, cmap = "gray")
plt.show()