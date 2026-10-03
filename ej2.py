import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import *

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

def ObtenerCaracteres(Casillero: np.ndarray, UmbralArea: int = 2) -> np.ndarray:
    ImagenUmbral = (Casillero < 130).astype(np.uint8)
    _, _, Stats, _ = cv2.connectedComponentsWithStats(ImagenUmbral, connectivity = 8)
    Stats = Stats[1:]
    Stats = Stats[Stats[:, cv2.CC_STAT_AREA] > UmbralArea]
    Stats = Stats[Stats[:, cv2.CC_STAT_WIDTH] < 0.5 * Casillero.shape[1]]
    return Stats[np.argsort(Stats[:, cv2.CC_STAT_LEFT])]

def ContarPalabras(Caracteres: np.ndarray, UmbralEspacio: int = 6) -> int:
    if len(Caracteres) == 0:
        return 0
    Izquierdas = Caracteres[:, cv2.CC_STAT_LEFT]
    Derechas = Izquierdas + Caracteres[:, cv2.CC_STAT_WIDTH]
    Huecos = Izquierdas[1:] - Derechas[:-1]
    return int(np.sum(Huecos > UmbralEspacio)) + 1

def ValidarNombre(Caracteres: np.ndarray) -> bool:
    return ContarPalabras(Caracteres) >= 2 and len(Caracteres) <= 12

def ValidarLegajo(Caracteres: np.ndarray) -> bool:
    return len(Caracteres) == 8 and ContarPalabras(Caracteres) == 1

def ValidarNota(Caracteres: np.ndarray) -> bool:
    return len(Caracteres) in (1, 2) and ContarPalabras(Caracteres) == 1

def ValidarCondicion(Caracteres: np.ndarray) -> bool:
    return len(Caracteres) == 1

def CaracteresCasillero(Imagen: np.ndarray, Casillero: Tuple[int,int,int,int]) -> np.ndarray:
    der, izq, sup, inf = Casillero
    return ObtenerCaracteres(Imagen[izq:der, sup:inf])

Validadores = [ValidarLegajo, ValidarNombre, ValidarNota, ValidarNota, ValidarNota, ValidarCondicion]
Campos = ["Legajo", "Nombre y apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]

def ValidarPlanilla(Imagen: np.ndarray) -> List[List[bool]]:
    Filas, Columnas = ObtenerGrilla(Imagen, 200)
    Casilleros = ObtenerCasilleros(Filas, Columnas)
    Resultados = []
    for Fila in range(1, 21):
        ResultadoFila = []
        print(f"> Registro {Fila}:")
        for Columna in range(1, 7):
            Caracteres = CaracteresCasillero(Imagen, Casilleros[Columna][Fila])
            EsValido = Validadores[Columna - 1](Caracteres)
            ResultadoFila.append(EsValido)
            print(f"> {Campos[Columna - 1]}: {'OK' if EsValido else 'MAL'}")
        print(">")
        Resultados.append(ResultadoFila)
    return Resultados

Imagen = cv2.imread("grade_sheet_3.png", cv2.IMREAD_GRAYSCALE)
Resultados = ValidarPlanilla(Imagen) 