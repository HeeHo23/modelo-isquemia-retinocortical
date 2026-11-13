"""
Script Principal: Executa todas as etapas e gera as figuras mínimas.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from src.malha_retinal import criar_malha_retinal
from src.mascaras_fisiologicas import (
    isquemia_retinal, mascara_foveal, atividade_retinal,
    converter_isquemia_para_oxigenio, mascara_acoplamento_oxigenio
)
from src.retinotopia import retina_para_cortex, cortex_para_retina
from src.simulacao_temporal import simular_progressao_temporal, medir_tunel
from src.campo_neural import partial_t_u, simular_fitzhugh_nagumo_cortical
from src.utils import normalizar01

os.makedirs("figuras", exist_ok=True)

# 1. Malha Retinal
N = 256
X, Y, R, Theta = criar_malha_retinal(N=N)

# 2. I_ret para 3 severidades + verificação centro/periferia
I_ret_padrao = isquemia_retinal(R, severidade=0.8, raio_preservado=0.25, largura_transicao=0.05)
print(f"[Bloco 2] Isquemia média centro (R<0.15): {I_ret_padrao[R < 0.15].mean():.4f}")
print(f"[Bloco 2] Isquemia média periferia (R>0.70): {I_ret_padrao[R > 0.70].mean():.4f}")

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, sev in zip(axs, [0.4, 0.7, 0.95]):
    im = ax.imshow(isquemia_retinal(R, severidade=sev), cmap="inferno", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"I_ret (severidade = {sev})")
    ax.axis("off")
plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8)
plt.savefig("figuras/01_mascara_isquemia_severidades.png", dpi=300)
plt.close()

# 3. F_ret e comparação de A_ret
F_ret = mascara_foveal(R, sigma_fovea=0.12)
print(f"[Bloco 3] F_ret max={F_ret.max():.2f}, min={F_ret.min():.4f}, centro={F_ret[N//2, N//2]:.4f}")

A_ret_basico = atividade_retinal(I_ret_padrao, F_ret=None, ganho_foveal=0.0)
A_ret_foveal = atividade_retinal(I_ret_padrao, F_ret=F_ret, ganho_foveal=0.35)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
axs[0].imshow(F_ret, cmap="viridis", origin="lower")
axs[0].set_title("Máscara Foveal (F_ret)")
axs[1].imshow(A_ret_basico, cmap="gray", origin="lower", vmin=0, vmax=1)
axs[1].set_title("A_ret: Perda Periférica Simples")
axs[2].imshow(A_ret_foveal, cmap="gray", origin="lower", vmin=0, vmax=1)
axs[2].set_title("A_ret: Perda Periférica + Ganho Foveal")
for ax in axs: ax.axis("off")
plt.tight_layout()
plt.savefig("figuras/02_preservacao_foveal_e_atividade.png", dpi=300)
plt.close()

# 5. Mapeamento Retino-Cortical (3 painéis)
A_cortex = retina_para_cortex(A_ret_foveal)
A_ret_reconstruida = cortex_para_retina(A_cortex)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
axs[0].imshow(A_ret_foveal, cmap="gray", origin="lower")
axs[0].set_title("Atividade Retinal (A_ret)")
axs[1].imshow(A_cortex, cmap="gray", origin="lower")
axs[1].set_title("Representação Cortical (A_cortex)")
axs[2].imshow(A_ret_reconstruida, cmap="gray", origin="lower")
axs[2].set_title("Reconstrução Perceptual")
for ax in axs: ax.axis("off")
plt.tight_layout()
plt.savefig("figuras/03_mapeamento_retino_cortical.png", dpi=300)
plt.close()

# 6 e 7. Progressão Temporal e Índice de Túnel
resultados_temp = simular_progressao_temporal(R, F_ret, T=6)
fig, axs = plt.subplots(1, 6, figsize=(16, 3))
medidas = []
for ax, res in zip(axs, resultados_temp):
    ax.imshow(res["A_rec"], cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"t = {res['t']}")
    ax.axis("off")
    m = medir_tunel(res["A_rec"], R)
    m["t"] = res["t"]
    medidas.append(m)
plt.tight_layout()
plt.savefig("figuras/04_progressao_temporal_tunel.png", dpi=300)
plt.close()

tempos = [m["t"] for m in medidas]
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(tempos, [m["contraste"] for m in medidas], marker="o", color="navy")
axs[0].set_title("Evolução do Contraste Centro-Periferia")
axs[0].set_xlabel("Tempo (t)"); axs[0].set_ylabel("Contraste")
axs[1].plot(tempos, [m["indice"] for m in medidas], marker="s", color="darkred")
axs[1].set_title("Evolução do Índice de Túnel")
axs[1].set_xlabel("Tempo (t)"); axs[1].set_ylabel("Índice de Túnel")
plt.tight_layout()
plt.savefig("figuras/05_metricas_contraste_tunel.png", dpi=300)
plt.close()

# 8. Grade 3x3 de Varredura de Parâmetros
severidades = [0.3, 0.6, 0.9]
raios = [0.15, 0.30, 0.45]
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for i, sev in enumerate(severidades):
    for j, raio in enumerate(raios):
        I_var = isquemia_retinal(R, severidade=sev, raio_preservado=raio)
        A_var = atividade_retinal(I_var, F_ret=F_ret, ganho_foveal=0.3)
        A_rec_var = cortex_para_retina(retina_para_cortex(A_var))
        axs[i, j].imshow(A_rec_var, cmap="gray", origin="lower", vmin=0, vmax=1)
        axs[i, j].set_title(f"sev={sev}, raio={raio}")
        axs[i, j].axis("off")
plt.tight_layout()
plt.savefig("figuras/06_varredura_parametros_3x3.png", dpi=300)
plt.close()

# 9. + Comparação das Máscaras de Acoplamento g(O) no Campo Neural
x_1d = np.linspace(-1, 1, 201)
R_1d = np.abs(x_1d)
I_1d = isquemia_retinal(R_1d, severidade=0.85, raio_preservado=0.25, largura_transicao=0.06)
F_1d = mascara_foveal(R_1d, sigma_fovea=0.12)
O_1d = converter_isquemia_para_oxigenio(I_1d, F_ret=F_1d, O_max=100.0, ganho_foveal_O2=0.15)

modos = ["linear", "alder_bifasico", "michaelis_menten", "drusas_faz"]
plt.figure(figsize=(9, 5))
for modo in modos:
    u_1d = 0.6 * np.exp(-(x_1d**2) / (2 * 0.25**2))  # Condição inicial u_0(x) Gaussiana (Chevallier et al., 2019)
    for _ in range(150):
        du = partial_t_u(u_1d, O_1d, x_1d, modo_oxigenio=modo, R=R_1d)
        u_1d += 0.05 * du
    plt.plot(x_1d, u_1d, linewidth=2, label=f"Modo g(O): {modo}")
plt.title("Equação de Campo Neural (Wilson-Cowan/Hawkes) sob Variabilidade de g(O)")
plt.xlabel("Posição Espacial Radial (x)"); plt.ylabel("Atividade Neural Estacionária u(t, x)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("figuras/07_comparacao_campo_neural_mascaras_O2.png", dpi=300)
plt.close()

# Reação-Difusão 2D (FitzHugh-Nagumo: Original vs Isquemia vs Isquemia + Ganho Foveal)
I_zero = np.zeros_like(R)
I_cortex_sem_fovea = retina_para_cortex(I_ret_padrao)
I_cortex_com_fovea = retina_para_cortex(np.clip(I_ret_padrao - 0.4 * F_ret, 0.0, 1.0))

u_orig, _, _ = simular_fitzhugh_nagumo_cortical(I_zero)
u_isq, _, _ = simular_fitzhugh_nagumo_cortical(I_cortex_sem_fovea)
u_fovea, _, _ = simular_fitzhugh_nagumo_cortical(I_cortex_com_fovea)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, u_map, tit in zip(axs, [u_orig, u_isq, u_fovea],
                          ["FHN Original", "FHN + Isquemia Cortical", "FHN + Isquemia + Ganho Foveal"]):
    rec = cortex_para_retina(normalizar01(u_map))
    ax.imshow(rec, cmap="magma", origin="lower")
    ax.set_title(tit); ax.axis("off")
plt.tight_layout()
plt.savefig("figuras/08_extensao_fitzhugh_nagumo_3casos.png", dpi=300)
plt.close()
print("Simulação concluída! Todas as figuras foram salvas na pasta 'figuras/'.")