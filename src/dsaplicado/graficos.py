"""Estilo gráfico común para todos los notebooks.

Paleta categórica validada para daltonismo. Los colores se asignan siempre en
el mismo orden y nunca se reciclan. Con más de tres series en un gráfico de
dispersión se usan etiquetas directas en vez de colores.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
SUPERFICIE = "#fcfcfb"
NEUTRO = "#b5b3ab"
CRITICO = "#d03b3b"

# Rampa secuencial azul de una sola tonalidad, de claro a oscuro.
AZUL_SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul_secuencial", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def aplicar_estilo() -> None:
    """Configura matplotlib con marcas finas, grilla discreta y tipografía del sistema."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "figure.dpi": 110,
            "figure.figsize": (10, 4.5),
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": TINTA,
            "axes.labelcolor": TINTA_SECUNDARIA,
            "axes.edgecolor": EJE,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=COLORES),
            "grid.color": GRILLA,
            "grid.linewidth": 0.6,
            "xtick.color": TINTA_TENUE,
            "ytick.color": TINTA_TENUE,
            "xtick.labelcolor": TINTA_SECUNDARIA,
            "ytick.labelcolor": TINTA_SECUNDARIA,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.labelcolor": TINTA_SECUNDARIA,
        }
    )


def eje_porcentaje(ax: plt.Axes, eje: str = "y", decimales: int = 0) -> None:
    """Formatea un eje con valores decimales como porcentaje."""
    formato = PercentFormatter(xmax=1.0, decimals=decimales)
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(formato)


def separar_etiquetas(posiciones: dict[str, float], separacion: float) -> dict[str, float]:
    """Ajusta posiciones verticales de etiquetas para que no se superpongan.

    Ordena las etiquetas por posición y empuja hacia arriba solo las que quedan a
    menos de ``separacion`` de la anterior. Las etiquetas aisladas no se mueven.
    """
    orden = sorted(posiciones.items(), key=lambda kv: kv[1])
    ajustadas: dict[str, float] = {}
    previa = None
    for nombre, valor in orden:
        pos = valor if previa is None else max(valor, previa + separacion)
        ajustadas[nombre] = previa = pos
    return ajustadas
