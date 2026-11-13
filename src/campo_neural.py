"""
Dinâmica Cortical - Reação-Difusão (FitzHugh-Nagumo) e Equação de Campo Neural
(Wilson-Cowan / Limite de Campo Médio de Hawkes) acoplada à máscara de oxigênio g(O).

Referências:
- Chevallier et al., 2019 (Eq. 1.2 e 3.23: Derivação rigorosa da equação
  de campo neural a partir de processos de Hawkes espacialmente estendidos)
- Pinotsis et al., 2014 (Eq. 8-10: Controle de ganho por contraste e dispersão espacial de conexões horizontais intrínsecas em V1)
"""

import numpy as np
from src.mascaras_fisiologicas import mascara_acoplamento_oxigenio

# =====================================================================
# PARTE A: Modelo de Campo Neural - Chevallier 2019 + Pinotsis 2014)
# =====================================================================
def f_wilson_cowan(u: np.ndarray, beta: float = 5.0, theta: float = 0.5) -> np.ndarray:
    """
    Função de taxa de disparo (firing rate function) sigmoidal deslocada com f(0) = 0.
    Satisfaz a Condição de Lipschitz (Assumption 1 de Chevallier et al., 2019) e equivale
    ao mapeamento não-linear F(v) de Pinotsis et al. (2014, Eq. 10), onde:
    - beta (float): Ganho sináptico / inclinação da sigmoide (r em Pinotsis et al., 2014).
    - theta (float): Limiar pós-sináptico de disparo (eta em Pinotsis et al., 2014).
    """
    return (
        1.0 / (1.0 + np.exp(-beta * (u - theta)))
        - 1.0 / (1.0 + np.exp(beta * theta))
    )

def w_conectividade(y: np.ndarray, x: np.ndarray, J: float = 1.0, sigma: float = 0.1,
                    tipo_kernel: str = "gaussiano") -> np.ndarray:
    """
    Kernel de força sináptica / conectividade espacial horizontal w(y, x).
    
    Parâmetros:
    - y, x (np.ndarray): Posições espaciais dos neurônios pré e pós-sinápticos.
    - J (float): Amplitude máxima do kernel de conectividade (a_ab em Pinotsis et al., 2014).
    - sigma (float): Dispersão espacial das conexões horizontais em V1 (inverso do decaimento
      c_ab em Pinotsis et al., 2014, que encolhe sob alto contraste/isquemia).
    - tipo_kernel (str): 'gaussiano' (import numpy as np.md) ou 'exponencial' (Pinotsis et al., Eq. 8).
    """
    dist = np.abs(x - y)
    if tipo_kernel == "gaussiano":
        return J * np.exp(-(dist**2) / (2.0 * sigma**2))
    elif tipo_kernel == "exponencial":
        # Kernel exponencial de cauda pesada usado em DCM para V1 (Pinotsis et al., 2014, Eq. 8)
        return J * np.exp(-dist / sigma)
    else:
        raise ValueError("tipo_kernel deve ser 'gaussiano' ou 'exponencial'.")

def partial_t_u(u: np.ndarray, O: np.ndarray, x: np.ndarray,
                alpha: float = 1.0, beta: float = 5.0, theta: float = 0.5,
                J: float = 1.0, sigma: float = 0.1, rho: np.ndarray = None,
                modo_oxigenio: str = "linear", **kwargs_oxigenio) -> np.ndarray:
    """
    Calcula a derivada temporal do potencial neural médio u(t, x) segundo a Equação de Campo Neural:
        du/dt(x) = -alpha * u(x) + integral w(y, x) * g(O(y)) * f(u(y)) * rho(y) dy
    
    Variáveis e Parâmetros:
    - u (np.ndarray): Estado neural atual u(t, x_i).
    - O (np.ndarray): Campo de oxigenação O(t, x_i) em [0, 100].
    - x (np.ndarray): Malha espacial 1D ou perfil radial [-1, 1].
    - alpha (float): Taxa de vazamento de membrana (leakage rate, Chevallier et al., 2019, Eq. 1.1).
    - rho (np.ndarray): Densidade empírica de distribuição espacial neuronal rho(dy) (Chevallier et al., 2019).
    - modo_oxigenio (str): Variação da máscara g(O) ('linear', 'alder_bifasico', 'michaelis_menten', 'drusas_faz').
    """
    if rho is None:
        rho = np.ones_like(x)
        
    y = x[:, None]
    xx = x[None, :]
    W = w_conectividade(y, xx, J=J, sigma=sigma)
    
    # Acoplamento de oxigênio g(O) selecionável conforme literatura
    g_O = mascara_acoplamento_oxigenio(O, modo=modo_oxigenio, **kwargs_oxigenio)
    
    # Atividade emitida em cada posição y
    firing = g_O * f_wilson_cowan(u, beta=beta, theta=theta) * rho
    
    # Integral espacial em y via regra do trapézio
    interaction = np.trapezoid(W * firing[:, None], x=x, axis=0)
    
    return -alpha * u + interaction


# =====================================================================
# PARTE B: Extensão de Reação-Difusão 2D (FitzHugh-Nagumo)
# =====================================================================
def laplaciano_2d(Z: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Calcula o operador Laplaciano discreto 2D por diferenças finitas com condições de contorno de Neumann.
    """
    Z_pad = np.pad(Z, pad_width=1, mode="edge")
    return (
        Z_pad[1:-1, 2:] + Z_pad[1:-1, :-2] +
        Z_pad[2:, 1:-1] + Z_pad[:-2, 1:-1] -
        4.0 * Z
    ) / (dx**2)

def simular_fitzhugh_nagumo_cortical(I_cortex: np.ndarray, k0: float = 0.15, alpha_isq: float = 0.12,
                                     D_u: float = 0.2, epsilon: float = 0.08, a_fhn: float = 0.7,
                                     b_fhn: float = 0.8, dt: float = 0.05, passos: int = 400) -> tuple:
    """
    Simula o modelo de Reação-Difusão de FitzHugh-Nagumo em coordenadas corticais:
        du/dt = D_u * laplaciano(u) + u - u^3 - v + K
        dv/dt = epsilon * (u + a_fhn - b_fhn * v)
    onde K = k0 - alpha_isq * I_cortex substitui o estímulo escalar por uma matriz espacial
    modulada pela isquemia cortical.
    """
    K = k0 - alpha_isq * I_cortex
    u = np.zeros_like(I_cortex, dtype=float)
    v = np.zeros_like(I_cortex, dtype=float)
    
    # Verificação explícita de compatibilidade de dimensões
    assert u.shape == v.shape == K.shape, "Dimensões de u, v e K devem ser idênticas!"
    
    for _ in range(passos):
        du = D_u * laplaciano_2d(u) + u - (u**3) - v + K
        dv = epsilon * (u + a_fhn - b_fhn * v)
        u += dt * du
        v += dt * dv
        
    return u, v, K