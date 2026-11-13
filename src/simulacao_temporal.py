"""
Progressão temporal da isquemia, métricas de contraste centro-periferia e varredura paramétrica.
Referências:  Böhm et al., 2023 (declínio linearde perfusão na progressão glaucomatosa/isquêmica).
"""

import numpy as np
from src.mascaras_fisiologicas import isquemia_retinal, atividade_retinal
from src.retinotopia import retina_para_cortex, cortex_para_retina

def simular_progressao_temporal(R: np.ndarray, F_ret: np.ndarray, T: int = 6,
                                severidade_max: float = 0.9, raio_inicial: float = 0.55,
                                raio_final: float = 0.15, largura_transicao: float = 0.05,
                                ganho_foveal: float = 0.3) -> list:
    """
    Simula a evolução discreta da isquemia em T passos de tempo.
    À medida que t avança, a severidade periférica cresce e o raio preservado encolhe.
    """
    resultados = []
    for t in range(T):
        frac = t / (T - 1) if T > 1 else 1.0
        severidade_t = severidade_max * frac
        raio_t = raio_inicial + (raio_final - raio_inicial) * frac
        
        I_ret_t = isquemia_retinal(
            R, severidade=severidade_t,
            raio_preservado=raio_t, largura_transicao=largura_transicao
        )
        A_ret_t = atividade_retinal(I_ret_t, F_ret=F_ret, ganho_foveal=ganho_foveal)
        A_cortex_t = retina_para_cortex(A_ret_t)
        A_rec_t = cortex_para_retina(A_cortex_t)
        
        resultados.append({
            "t": t,
            "frac": frac,
            "severidade": severidade_t,
            "raio_preservado": raio_t,
            "I_ret": I_ret_t,
            "A_ret": A_ret_t,
            "A_cortex": A_cortex_t,
            "A_rec": A_rec_t
        })
    return resultados

def medir_tunel(A: np.ndarray, R: np.ndarray, r_centro: float = 0.15, r_periferia: float = 0.65) -> dict:
    """
    Calcula o contraste centro-periferia e o índice de visão em túnel ("luz no fim do túnel").
    
    Variáveis:
    - r_centro (float): Limite radial considerado como centro foveal/parafoveal (R < 0.15).
    - r_periferia (float): Limite radial considerado como periferia (R > 0.65).
    """
    mascara_centro = R < r_centro
    mascara_periferia = R > r_periferia
    
    media_centro = float(A[mascara_centro].mean())
    media_periferia = float(A[mascara_periferia].mean())
    contraste = media_centro - media_periferia
    indice = media_centro / (media_periferia + 1e-8)
    
    return {
        "media_centro": media_centro,
        "media_periferia": media_periferia,
        "contraste": contraste,
        "indice": indice
    }