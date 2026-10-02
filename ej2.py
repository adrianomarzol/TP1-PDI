import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import *

Imagen = cv2.imread("grade_sheet_1.png", cv2.IMREAD_GRAYSCALE)

def ObtenerGrilla(Imagen: np.ndarray, Umbral: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detecta las filas y columnas de la grilla de una imagen.

    Recibe:
    Imagen: array de una imagen con una grilla.
    Umbral: entero para umbralizar la imagen.

    Devuelve:
    IdFilasGrilla: array con indices de las filas de la grilla.
    IdColumnasGrilla: array con indices de las columnas de la grilla.
    """
    ImagenUmbralada = (Imagen < Umbral).astype(np.uint8)
    Alto, Ancho = ImagenUmbralada.shape

    FilasImagen = ImagenUmbralada.sum(axis = 1)
    ColumnasImagen = ImagenUmbralada.sum(axis = 0)

    FilasGrilla = FilasImagen > 0.7 * Alto
    ColumnasGrilla = ColumnasImagen > 0.5 * Ancho

    FilasGrilla = np.diff(FilasGrilla)
    IdFilasGrilla = np.argwhere(FilasGrilla)
    ColumnasGrilla = np.diff(ColumnasGrilla)
    IdColumnasGrilla = np.argwhere(ColumnasGrilla)
    
    return IdFilasGrilla, IdColumnasGrilla

def ObtenerCasilleros(IdFilas: np.ndarray, IdColumnas: np.ndarray) -> List[List[Tuple[int,int,int,int]]]:
    """
    """
    CasillerosColumna = []
    for I in range(1, len(IdColumnas) - 1, 2):
        Casilleros = []
        for J in range(1, len(IdFilas) - 1, 2):
            BordeSuperior = IdColumnas[I][0]
            BordeInferior = IdColumnas[I+1][0]

            BordeIzquierdo = IdFilas[J][0]
            BordeDerecho = IdFilas[J+1][0]

            Casilleros.append((BordeDerecho, BordeIzquierdo, BordeSuperior, BordeInferior))

        CasillerosColumna.append(Casilleros)

    return CasillerosColumna

def ContarCaracteresPalabras(Casillero):
    return 0
    

F, C = ObtenerGrilla(Imagen, 200)

Casilleros = ObtenerCasilleros(F, C)

der, izq, sup, inf = Casilleros[2][1]

plt.plot(C)
plt.show()

Casilleros[0]
Casillero = Imagen[izq:der, sup:inf]

ImagenUmbral = (Casillero < 180).astype(np.uint8)

plt.imshow(ImagenUmbral, cmap = 'gray')
plt.show()

nl, l, stats, centroids = cv2.connectedComponentsWithStats(ImagenUmbral, connectivity = 8)

ix_area = stats[:, -1] > 50
stats = stats[ix_area, :] 

for i in range(1, nl):
    # Extraer estadísticas de la componente actual
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    
    # Extraer las coordenadas del centroide
    cx, cy = centroids[i]
    
    # Opcional: Filtrar por área para ignorar ruido (ej. ignorar cosas de menos de 20 píxeles)
    if area > 100:
        # Dibujar el rectángulo verde alrededor del objeto (espesor de 2 píxeles)
        cv2.rectangle(ImagenUmbral, (x, y), (x + w, y + h), 2)
        
        # Dibujar un círculo rojo en el centroide (radio 3, relleno)
        #cv2.circle(ImagenUmbral[izq:der, sup:inf], (int(cx), int(cy)), 3, (0, 0, 255), -1)

# 5. Mostrar el resultado en una ventana
plt.imshow(ImagenUmbral, cmap = 'gray')
plt.show()