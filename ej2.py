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
    Devuelve los casilleros de la grilla a partir de los indices de filas y columnas.
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
def GuardarCSV(Resultados: List[List[bool]], Ruta: str) -> None:
    import csv
    with open(Ruta, "w", newline="", encoding="utf-8-sig") as Archivo:
        Escritor = csv.writer(Archivo)
        Escritor.writerow(["ID"] + Campos)
        for Id, ResultadoFila in enumerate(Resultados, start=1):
            Escritor.writerow([Id] + ["OK" if Valido else "MAL" for Valido in ResultadoFila])

Resultados = ValidarPlanilla(Imagen)
GuardarCSV(Resultados, "resultados.csv")

def ClasificarCondicion(Imagen: np.ndarray, Casillero: Tuple[int,int,int,int]) -> str:
    Caracteres = CaracteresCasillero(Imagen, Casillero)
    Area = Caracteres[0][cv2.CC_STAT_AREA]
    Ancho = Caracteres[0][cv2.CC_STAT_WIDTH]
    Alto = Caracteres[0][cv2.CC_STAT_HEIGHT]
    if Area / (Ancho * Alto) >= 0.40:
        return "R"
    if Ancho / Alto >= 0.8:
        return "A"
    return "L"

Filas, Columnas = ObtenerGrilla(Imagen, 200)
Casilleros = ObtenerCasilleros(Filas, Columnas)
Recortes = []
for Fila in range(1, 21):
    if all(Resultados[Fila - 1]):
        Condicion = ClasificarCondicion(Imagen, Casilleros[6][Fila])
        if Condicion in ("L", "R"):
            der, izq, sup, inf = Casilleros[2][Fila]
            Recorte = cv2.cvtColor(Imagen[izq:der, sup:inf], cv2.COLOR_GRAY2BGR)
            Color = (0, 0, 255) if Condicion == "L" else (255, 0, 0)
            Recorte = cv2.copyMakeBorder(Recorte, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=Color)
            Recortes.append(Recorte)

if len(Recortes) > 0:
    cv2.imwrite("alumnos_no_aprobados.png", np.vstack(Recortes))

for id_planilla in range(1, 5):
    nombre_archivo = f"grade_sheet_{id_planilla}.png"
    img_actual = cv2.imread(nombre_archivo, cv2.IMREAD_GRAYSCALE)
    if img_actual is None:
        continue
    res= ValidarPlanilla(img_actual)
    GuardarCSV(res, f"resultados_{nombre_archivo[:-4]}.csv")
    f_grilla, c_grilla = ObtenerGrilla(img_actual, 200)
    casilleros_actual = ObtenerCasilleros(f_grilla, c_grilla)
    recortes_no_aprobados = []
    for fila in range(1, 21):
        if all(res[fila - 1]):
            cond = ClasificarCondicion(img_actual, casilleros_actual[6][fila])
            if cond in ("L", "R"):
                der, izq, sup, inf = casilleros_actual[2][fila]
                rec = cv2.cvtColor(img_actual[izq:der, sup:inf], cv2.COLOR_GRAY2BGR)
                color = (0, 0, 255) if cond == "L" else (255, 0, 0)
                rec = cv2.copyMakeBorder(rec, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
                recortes_no_aprobados.append(rec)

    if len(recortes_no_aprobados) > 0:
        cv2.imwrite(f"alumnos_no_aprobados_{id_planilla}.png", np.vstack(recortes_no_aprobados))