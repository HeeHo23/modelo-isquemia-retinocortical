"""
Funções auxiliares de normalização e visualização de matrizes.
"""

import numpy as np
import matplotlib.pyplot as plt

def normalizar01(A: np.ndarray) -> np.ndarray:
    """
    Normaliza uma matriz numérica para o intervalo fechado [0,1].

    Variáveis:
    Matriz de entrada A (ex: R, A_cortex, u)
    - amin, amax (float): Valores mínimo e máximo da matriz A.
    - 1e-12 (float): Constante de regularização numérica (evita divisão por zero
      caso a matriz seja uniforme, conforme discutido em Sacco et al., 2019, Cap. 1).
    """
    A = A.astype(float)
    amin = A.min()
    amax = A.max()
    return (A - amin) / (amax - amin + 1e-12)

def mostrar_imagem(A: np.ndarray, titulo: str = "", cmap: str = "gray", salvar_caminho: str = None):
    """
    Exibe uma matriz como uma imagem.

    Variáveis:
    Matriz de entrada A (ex: R, A_cortex, u)
    - titulo (str): Título da imagem.
    - cmap (str): Colormap para a imagem (padrão: "gray").
    - salvar_caminho (str): Caminho para salvar a imagem (opcional).
    """
    plt.figure(figsize=(5,5))
    plt.imshow(A, cmap=cmap, origin="lower")
    plt.title(titulo)
    plt.axis('off')
    plt.tight_layout()
    if salvar_caminho:
        plt.savefig(salvar_caminho, dpi = 300)
    plt.show()