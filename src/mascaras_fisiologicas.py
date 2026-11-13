"""
Máscaras de isquemia retinal, preservação foveal, atividade retinal e variabilidade das máscaras de acoplamento de oxigênio.
Referências:
- Alder & Constable, 1981 [450 (1).pdf] (Resposta bifásica de células ganglionares à hipóxia)
- Ernest & Krill, 1971 [Central retina versus peripheral__retina.pdf] (Vulnerabilidade periférica 45° vs central 5°)
- Guidoboni et al., 2019 [978-3-030-25886-3.pdf] (Cinética de Michaelis-Menten e Krogh)
- McHugh et al., 2019 [file (3).pdf] (Hipóxia por drusas > 64um e função degrau)
- Yoshimura et al., 2024 (Geometria da FAZ e inibição astrocitária)
"""

import numpy as np
# =====================================================================
# Isquemia Retinal Periférica
# =====================================================================
def isquemia_retinal(R: np.ndarray, severidade: float = 0.8, 
                     raio_preservado: float = 0.25, largura_transicao: float = 0.05) -> np.ndarray:
    """
    Gera a matriz de isquemia retinal periférica I_ret em [0, 1].
    Reflete o achado de Ernest & Krill (1971): a retina periférica (45° de excentricidade,
    suprida pela artéria central da retina com menor PO2 distal) sofre maior elevação de
    limiar visual sob hipóxia do que a retina central (5°).
    
    Parâmetros:
    - R (np.ndarray): Matriz de distância radial normalizada [0, 1].
    - severidade (float): Intensidade máxima da isquemia na periferia [0, 1].
    - raio_preservado (float): Raio central poupado pela isquemia (ex: 0.25).
    - largura_transicao (float): Suavidade da transição sigmoidal centro-periferia.
    
    Retorna:
    - I_ret (np.ndarray): 0 = retina preservada, 1 = retina fortemente comprometida.
    """
    I_ret = severidade / (1.0 + np.exp(-(R - raio_preservado) / largura_transicao))
    return np.clip(I_ret, 0.0, 1.0)

# =====================================================================
# Preservação Foveal
# =====================================================================
def mascara_foveal(R: np.ndarray, sigma_fovea: float = 0.12) -> np.ndarray:
    """
    Gera a máscara gaussiana F_ret centrada na fóvea.
    Fisiologicamente, a fóvea (FAZ) é avascular na retina interna, mas 85-90% de seu O2
    vem por difusão direta da coriocapilar hiperperfundida (McHugh et al., 2019;
    Prada et al., 2019 em Ocular Fluid Dynamics), conferindo preservação relativa.
    
    Parâmetros:
    - R (np.ndarray): Distância radial [0, 1].
    - sigma_fovea (float): Desvio padrão espacial que controla o raio foveal preservado
      (análogo ao raio d_0 da FAZ em Yoshimura et al., 2024).
    """
    F_ret = np.exp(-(R**2) / (2.0 * sigma_fovea**2))
    return F_ret / F_ret.max()

# =====================================================================
# Atividade Retinal Simulada
# =====================================================================
def atividade_retinal(I_ret: np.ndarray, F_ret: np.ndarray = None, ganho_foveal: float = 0.0) -> np.ndarray:
    """
    Constrói a atividade visual retinal simulada A_ret a partir de uma luminância
    basal uniforme L_ret reduzida pela isquemia e amplificada pelo ganho foveal.
    """
    L_ret = np.ones_like(I_ret)
    A_ret = L_ret * (1.0 - I_ret)
    if F_ret is not None:
        A_ret = A_ret + ganho_foveal * F_ret
    return np.clip(A_ret, 0.0, 1.0)

# =====================================================================
# VARIABILIDADE DAS MÁSCARAS DE ACOPLAMENTO DE OXIGÊNIO g(O)
# =====================================================================
def converter_isquemia_para_oxigenio(I_ret: np.ndarray, F_ret: np.ndarray = None,
                                     O_max: float = 100.0, ganho_foveal_O2: float = 0.2) -> np.ndarray:
    """
    Converte a máscara adimensional I_ret em um campo espacial de oxigenação O(x)
    em porcentagem [0, 100]% ou pressão parcial PO2 [0, 80] mmHg (McHugh et al., 2019).
    """
    O_field = O_max * (1.0 - I_ret)
    if F_ret is not None:
        O_field += O_max * ganho_foveal_O2 * F_ret
    return np.clip(O_field, 0.0, O_max)

def mascara_acoplamento_oxigenio(O: np.ndarray, modo: str = "linear", **kwargs) -> np.ndarray:
    """
    Implementa a variabilidade da função de acoplamento de oxigênio g(O) que modula
    a atividade neural / taxa de disparo (firing rate) conforme os PDFs de referência.
    
    Parâmetros:
    - O (np.ndarray): Campo espacial de oxigenação (0 a 100%).
    - modo (str): Seleciona a filosofia biofísica de acoplamento:
        1. 'linear': Modelo base de `import numpy as np.md` (g(O) = O / 100).
        2. 'alder_bifasico': Baseado em Alder & Constable (1981).
           Modela o platô de autorregulação (>45% / 45 mmHg), o pico transitório de
           hiperexcitabilidade (MFR aumentada em 67% das células ganglionares entre
           24% e 45%) e o colapso isquêmico (<21%).
        3. 'michaelis_menten': Baseado em Arciero et al. (2019)
           e McHugh et al. (2019). Usa saturação enzimática com corte
           em zero (step function) para anóxia.
        4. 'drusas_faz': Combina a proteção foveal da FAZ (Yoshimura et al., 2024) com
           defeitos focais de difusão por drusas > 64 um (McHugh et al., 2019).
    """
    O_clamped = np.maximum(O, 0.0)  # Step function física: evita O2 < 0 (McHugh et al., 2019)

    if modo == "linear":
        # Referência: import numpy as np.md
        return O_clamped / 100.0

    elif modo == "alder_bifasico":
        # Referência: Alder & Constable (1981), Fig. 2-3 e Tabela I.
        # PO2 > 45 mmHg: função normal (g = 1.0)
        # 24 <= PO2 <= 45 mmHg: aumento paradoxal da MFR (overshoot/hiperexcitabilidade)
        # PO2 < 21 mmHg: cessação completa de disparo (g -> 0)
        limiar_critico = kwargs.get("limiar_critico", 22.0)
        incl_queda = kwargs.get("incl_queda", 4.0)
        centro_pico = kwargs.get("centro_pico", 32.0)
        largura_pico = kwargs.get("largura_pico", 6.0)
        amp_hiper = kwargs.get("amp_hiper", 0.35)  # +35% de disparo na zona de hipóxia moderada

        # Termo 1: Queda sigmoidal abrupta abaixo de 24 mmHg
        base_sigmoide = 1.0 / (1.0 + np.exp(-(O_clamped - limiar_critico) / incl_queda))
        # Termo 2: Bump gaussiano de hiperexcitabilidade entre 24 e 45 mmHg
        bump_hipoxia = amp_hiper * np.exp(-((O_clamped - centro_pico)**2) / (2.0 * largura_pico**2))
        
        return np.clip(base_sigmoide + bump_hipoxia, 0.0, 1.5)

    elif modo == "michaelis_menten":
        # Referência: Arciero et al. (2019) Cap. 5 / McHugh et al. (2019)
        # K_m é a tensão de O2 onde o consumo/acoplamento cai pela metade (tipicamente ~2 a 5 mmHg)
        K_m = kwargs.get("K_m", 5.0)
        limiar_anoxia = kwargs.get("limiar_anoxia", 1.0)
        g_val = O_clamped / (O_clamped + K_m)
        # Aplica step function para anóxia estrita
        g_val[O_clamped < limiar_anoxia] = 0.0

        return g_val / (100.0 / (100.0 + K_m))  # Normalizado para g(100) = 1.0

    elif modo == "drusas_faz":
        # Referência: McHugh et al. (2019) - Drusas > 64 um geram ilhas de anóxia local
        R_grid = kwargs.get("R", np.zeros_like(O_clamped))
        # Cria atenuação localizada simulando barreira difusiva de drusas perifoveais (R ~ 0.2)
        barreira_drusa = 1.0 - 0.85 * np.exp(-((R_grid - 0.22)**2) / (2.0 * 0.03**2))
        O_efetivo = O_clamped * barreira_drusa
        
        return O_efetivo / 100.0

    else:
        raise ValueError(f"Modo '{modo}' desconhecido para a máscara de acoplamento.")