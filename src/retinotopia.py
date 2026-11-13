"""
Mapeamento retino-cortical e reconstrução perceptual via transformação Log-Polar.
"""

import numpy as np
import cv2
from src.utils import normalizar01

def preparar_imagem_cv2(A:np.darray) -> np.ndarray:
    """
    Converte matriz float [0,1] para uint8 [0,255] exigido pelo OpenCV (cv2) para operações de interpolação
    """
    A01 = normalizar01(A)
    A8 = (255.0 * A01).astype(np.uint8)
    return A8

def retina_para_cortex(A_ret: np.ndarray, radius: float = None) -> np.ndarray:
    """
    Transforma a atividade do domínio retinal (polar/radial) para o domínio cortical V1
    (cartesiano log-polar). Na retinotopia biológica, a fóvea sofre magnificação cortical.
    
    Ponto técnico: center deve ser passado como (w/2, h/2).
    """
    A8 = preparar_imagem_cv2(A_ret)
    h, w = A8.shape[:2]
    if radius is None:
        radius = min(h, w) / 2.0
    center = (w / 2.0, h / 2.0)
    
    A_cortex = cv2.warpPolar(
        A8,
        dsize=(w, h),
        center=center,
        maxRadius=radius,
        flags=cv2.WARP_POLAR_LOG
    )
    return normalizar01(A_cortex)

def cortex_para_retina(A_cortex: np.ndarray, radius: float = None) -> np.ndarray:
    """
    Transformação inversa (córtex V1 -> reconstrução perceptual no campo visual).
    Usa a flag combinada WARP_POLAR_LOG + WARP_INVERSE_MAP.
    """
    A8 = preparar_imagem_cv2(A_cortex)
    h, w = A8.shape[:2]
    if radius is None:
        radius = min(h, w) / 2.0
    center = (w / 2.0, h / 2.0)
    
    A_ret_rec = cv2.warpPolar(
        A8,
        dsize=(w, h),
        center=center,
        maxRadius=radius,
        flags=cv2.WARP_POLAR_LOG + cv2.WARP_INVERSE_MAP
    )
    return normalizar01(A_ret_rec)