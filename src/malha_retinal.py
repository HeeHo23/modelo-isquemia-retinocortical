"""
Construção da malha retinal em coordenadas cartesianas e polares.
"""

import numpy as np

def criar_malha_retinal(N: int=512):
    """
    Cria uma grade quadrada NxN representando o campo visual centrada na fóvea.
    
    Parâmetros:
    N (int): Resolução espacial da grade (padrão 512 x 512 pixels)
    
    Variáveis retornadas:
    - X, Y (np.ndarray): Coordenadas cartesianas da grade [0, N-1].
    - R (np.ndarray): Distância radial euclidiana normalizada em [0,1].
    R=0 representa a fóvea e R=1 representa a periferia extrema (ora serrata / 45º de excentricidade, cf. Ernest & Krill,1971)"
    - Theta (np.ndarray): Ângulo polar em radianos [-pi, pi], útil para modelar
      assimetrias anatômicas (ex: rafe horizontal temporal vs. disco óptico nasal,
      cf. Yoshimura et al., 2024).
    """
    Y,X = np.mgrid[0:N, 0:N]
    cx = (N-1) / 2.0 # Coordenada X do centro foveal
    cy = (N-1) / 2.0 # Coordenada Y do centro foveal

    # Distância radial ao ponto de fixação
    R = np.sqrt((X-cx)**2 + (Y-cy)**2)
    R = R/R.max() # Normalização para [0,1]

    # Ângulo polar de cada ponto da retina
    Theta = np.arctan2 (Y - cy, X-cx)

    return X, Y, R, Theta