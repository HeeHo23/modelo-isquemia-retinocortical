# Modelo Computacional Retino-Cortical da Visão em Túnel sob Isquemia e Hipóxia

Este repositório implementa um modelo computacional exploratório, desenvolvido de forma modular e versionada em etapas, para investigar como alterações isquêmicas e hipóxicas periféricas da retina — combinadas com a preservação metabólica foveal, o mapeamento retinotópico log-polar e a dinâmica de campos neurais em V1 — podem gerar padrões perceptuais compatíveis com o fenômeno de **"luz no fim do túnel"**.

> **Escopo.** O modelo é fenomenológico: os parâmetros foram escolhidos por plausibilidade qualitativa e **não foram ajustados a dados experimentais**. As seções 3 e 4 distinguem o que a implementação efetivamente produz do que é apenas hipótese.

---

## 1. Introdução

A retina possui uma das maiores demandas metabólicas por grama de tecido do corpo humano, tornando a função visual altamente vulnerável a reduções na pressão de perfusão ocular (OPP) e na oferta de oxigênio (hipóxia/isquemia). O suprimento sanguíneo retinal é organizado de maneira dual e espacialmente heterogênea: a retina interna é nutrida pelos plexos capilares superficial, intermediário e profundo derivados da artéria central da retina (CRA), enquanto a retina externa e a zona avascular foveal (FAZ) dependem majoritariamente (cerca de 85–90% na fóvea) da difusão de oxigênio a partir da coriocapilar.

Estudos psicofísicos clássicos (Ernest & Krill, 1971) indicam que a hipóxia sistêmica ($10\%\ \text{O}_2$ inspirado) eleva os limiares visuais de forma mais severa na periferia retinal ($45^\circ$ de excentricidade) do que na região central ($5^\circ$). Adicionalmente, durante a ontogenia da arquitetura macular humana, a migração de astrócitos guiada por fatores repulsivos foveais molda a FAZ (Yoshimura et al., 2024).

No domínio eletrofisiológico, gravações extracelulares em células ganglionares da retina (RGCs) (Alder & Constable, 1981) sugerem uma resposta não-linear à hipóxia: taxa de disparo mantida (MFR) estável para $PO_2$ arterial $> 45\ \text{mmHg}$, aumento transitório (hiperexcitabilidade) entre $24$ e $45\ \text{mmHg}$ e supressão até a cessação completa abaixo de $21\ \text{mmHg}$. Os sinais aferentes projetam-se para V1 por uma transformação retinotópica log-polar que magnifica a representação foveal. Em V1, interações horizontais intrínsecas controlam o ganho e a somação espacial (Pinotsis et al., 2014), e a dinâmica populacional pode ser descrita, no limite de campo médio de processos de Hawkes espacialmente estendidos (Chevallier et al., 2019), por uma equação integro-diferencial de campo neural do tipo Wilson-Cowan.

O objetivo do projeto é integrar esses mecanismos em um laboratório virtual reprodutível e explorar em quais regimes de isquemia periférica e ganho foveal emerge a reconstrução perceptual de visão em túnel, e como diferentes formulações do acoplamento de oxigênio $g(O)$ alteram um campo neural 1D.

---

## 2. Métodos

Cada conceito fisiológico é mapeado para um objeto computacional. O pipeline 2D de `main.py` usa uma grade $N \times N$ com $N = 256$ (o padrão da função `criar_malha_retinal` é $N = 512$, mas `main.py` fixa 256).

### 2.1. Malha Retinal e Máscaras Fisiológicas (`v0.1.0` e `v0.2.0`)

* **Geometria (`src/malha_retinal.py`)**: o centro $(c_x, c_y) = ((N-1)/2,\ (N-1)/2)$ define o ponto de fixação. A distância euclidiana ao centro é **dividida pelo seu máximo**, que ocorre nos *cantos* da imagem; logo $R \in [R_{\min}, 1]$ com $R_{\min} = 1/255 \approx 0{,}004$ (não há pixel exatamente em $R=0$ com $N$ par). O ponto médio de cada borda tem $R \approx 0{,}707$; apenas os cantos atingem $R = 1$. A matriz $\Theta = \text{arctan2}(Y - c_y,\ X - c_x) \in [-\pi, \pi]$ é gerada, mas **não é usada** nos cálculos atuais.
* **Isquemia Retinal Periférica ($I_{ret}$)** — sigmoide radial crescente:
  $$I_{ret}(R) = \text{clip}\left( \frac{s}{1 + \exp\left(-\frac{R - r_{p}}{\Delta r}\right)}, 0, 1 \right)$$
  com severidade $s$ (padrão $0{,}8$), raio preservado $r_p$ (padrão $0{,}25$) e largura $\Delta r$ (padrão $0{,}05$). Note que $I_{ret}$ no centro **não é exatamente zero** ($\approx 0{,}005\,s/0{,}8$) e satura em $s$, não em 1, quando $s<1$.
* **Preservação Foveal ($F_{ret}$)**: $F_{ret}(R) = \exp(-R^2/2\sigma_{\text{fovea}}^2)$, dividido pelo seu máximo (padrão $\sigma_{\text{fovea}}=0{,}12$). Para $R>0{,}35$, $F_{ret} < 0{,}015$.
* **Atividade Retinal ($A_{ret}$)**: a luminância basal é uniforme, $L_{ret}\equiv 1$:
  $$A_{ret}(R) = \text{clip}\left( L_{ret}\,(1 - I_{ret}) + g_{\text{fov}}\, F_{ret},\ 0,\ 1 \right)$$
  Como $1 - I_{ret} \approx 1$ no centro, o ganho foveal só tem efeito visível na região de transição ($R \approx 0{,}15$–$0{,}35$); no centro, o `clip` já satura em 1 (ver 3.2).

### 2.2. Variantes da função de acoplamento de oxigênio $g(O)$ (`v0.2.0`)

O campo de oxigenação é construído por `converter_isquemia_para_oxigenio`:
$$O(x) = \text{clip}\big(O_{\max}(1 - I_{ret}) + O_{\max}\, g_{O_2}\, F_{ret},\ 0,\ O_{\max}\big),\quad O_{\max}=100,$$
e tratado como porcentagem em $[0,100]$. **Os limiares fisiológicos em mmHg (21, 24, 45) são usados diretamente como valores nessa escala**, isto é, assume-se $O\,[\%] \equiv PO_2\,[\text{mmHg}]$ — uma simplificação, não uma conversão calibrada.

As quatro variantes de `mascara_acoplamento_oxigenio` (`src/mascaras_fisiologicas.py`), todas com $O \leftarrow \max(O, 0)$:

1. **`linear`**: $g(O) = O/100$.
2. **`alder_bifasico`** (inspirado em Alder & Constable, 1981):
   $$g(O) = \text{clip}\!\left(\frac{1}{1+e^{-(O-22)/4}} + 0{,}35\, e^{-\frac{(O-32)^2}{2\cdot 6^2}},\ 0,\ 1{,}5\right)$$
   Não há platô exato em 1 para $O>45$ (a sigmoide aproxima 1 assintoticamente); o pico de hiperexcitabilidade tem centro em 32 e vale $g\approx 1{,}27$ ($g(45)\approx 1{,}03$; $g(22)\approx 0{,}59$; $g(10)\approx 0{,}05$).
3. **`michaelis_menten`** (Arciero et al., 2019 e McHugh et al., 2019, conforme os comentários do código): $g(O)=\dfrac{O/(O+K_m)}{100/(100+K_m)}$ com $K_m = 5$, normalizado para $g(100)=1$, e $g=0$ para $O<1$ (corte de anóxia). Trata-se de uma **saturação suave**: $g(10)\approx 0{,}70$, $g(22)\approx 0{,}86$; não há queda abrupta em $K_m$.
4. **`drusas_faz`**: $g(O)=\dfrac{O\,[\,1 - 0{,}85\,e^{-(R-0{,}22)^2/(2\cdot 0{,}03^2)}\,]}{100}$, ou seja, uma **atenuação gaussiana localizada em $R = 0{,}22$** (largura $0{,}03$) sobre o acoplamento linear, usada como proxy de barreira difusiva por drusas ($>64\,\mu\text{m}$). O modelo **não** implementa explicitamente a geometria da FAZ nem barreiras anulares.

### 2.3. Mapeamento Retino-Cortical e Simulação Temporal (`v0.3.0` e `v0.4.0`)

* **Retinotopia (`src/retinotopia.py`)**: `retina_para_cortex` converte a matriz para `uint8` após **normalização min–max** (`normalizar01`), aplica `cv2.warpPolar` com `WARP_POLAR_LOG` (centro $(w/2, h/2)$, `maxRadius = min(h,w)/2`) e normaliza novamente. `cortex_para_retina` usa `WARP_POLAR_LOG + WARP_INVERSE_MAP`. Consequências importantes:
  * `maxRadius = N/2` corresponde a $R \approx 0{,}707$; **os cantos ($R>0{,}707$) ficam fora do mapa** e não são reconstruídos.
  * A normalização min–max a cada etapa **descarta a escala absoluta** e, para imagens uniformes (ex.: $t=0$ na seção 2.3), produz imagem nula.
  * A quantização para 8 bits e a interpolação tornam o ciclo retina → córtex → retina **com perdas**.
* **Progressão Temporal (`src/simulacao_temporal.py`)**: $T = 6$ estágios, com $f = t/(T-1)$: severidade $s_t = 0{,}9\,f$ (de $0$ a $0{,}9$), raio preservado $r_t = 0{,}55 - 0{,}40\,f$ (de $0{,}55$ a $0{,}15$), $\Delta r = 0{,}05$, $\sigma_{\text{fovea}} = 0{,}12$ e $g_{\text{fov}} = 0{,}3$.
* **Métricas (`medir_tunel`)**, calculadas sobre a **reconstrução** $A_{rec}$ (não sobre $A_{ret}$): contraste $C = \bar{A}_{R<0{,}15} - \bar{A}_{R>0{,}65}$ e índice $I_{\text{túnel}} = \bar{A}_{R<0{,}15}/(\bar{A}_{R>0{,}65} + 10^{-8})$. A região $R>0{,}65$ inclui a coroa $0{,}65<R<0{,}707$ e os cantos, que estão fora do mapa log-polar.

### 2.4. Dinâmica Cortical (`v1.0.0`, `src/campo_neural.py`)

1. **Equação de campo neural (inspirada em Chevallier et al., 2019 e Pinotsis et al., 2014)** — resolvida em um **perfil radial 1D**, $x\in[-1,1]$ (201 pontos), **desacoplado do pipeline 2D**:
   $$\frac{\partial u(t, x)}{\partial t} = -\alpha\, u(t, x) + \int_{\Omega} w(y, x)\, g(O(y))\, f(u(t, y))\, \rho(y)\, dy$$
   com $f(u) = \dfrac{1}{1 + e^{-\beta(u - \theta)}} - \dfrac{1}{1 + e^{\beta\theta}}$ ($f(0)=0$), $w(y,x) = J\,e^{-(x-y)^2/2\sigma^2}$ (há também a opção `exponencial`, não usada em `main.py`), $\alpha=1$, $\beta=5$, $\theta=0{,}5$, $J=1$, $\sigma=0{,}1$ e $\rho\equiv 1$. A integral é calculada pela regra do trapézio. Como $f(0)=0$, **$u=0$ é ponto fixo**.
   Em `main.py`: $u_0(x)=0{,}6\,e^{-x^2/(2\cdot0{,}25^2)}$, integração por Euler explícito, $\Delta t = 0{,}05$, 150 passos ($t=7{,}5$), e $O(x)$ gerado com $s=0{,}85$, $r_p=0{,}25$, $\Delta r=0{,}06$, $\sigma_{\text{fovea}}=0{,}12$, $g_{O_2}=0{,}15$.
2. **Reação-Difusão 2D de FitzHugh-Nagumo** (matriz cortical $256\times256$):
   $$\frac{\partial u}{\partial t} = D_u \nabla^2 u + u - u^3 - v + K, \qquad \frac{\partial v}{\partial t} = \epsilon\,(u + a - b\,v)$$
   com $K = k_0 - \alpha_{\text{isq}}\, I_{\text{cortex}}$, $k_0=0{,}15$, $\alpha_{\text{isq}}=0{,}12$, $D_u=0{,}2$, $\epsilon=0{,}08$, $a=0{,}7$, $b=0{,}8$, Euler explícito com $\Delta t=0{,}05$, 400 passos ($t=20$), condição inicial $u=v=0$ e contorno de Neumann (padding por réplica de borda). Como $I_{\text{cortex}}$ vem de `retina_para_cortex` (com min–max), sua escala é **renormalizada para $[0,1]$**, e $K\in[0{,}03;\,0{,}15]$.

---

## 3. Resultados

### 3.1. Guia de escalas, colormaps e convenções visuais

| Figura | Colormap | Escala | Leitura |
| :--- | :--- | :--- | :--- |
| **`01` — $I_{ret}$** | `inferno` | $[0,1]$ fixa (`vmin=0, vmax=1`) | Preto: preservado; amarelo: isquemia máxima (limitada à severidade do painel). |
| **`02` — $F_{ret}$** | `viridis` | **autoescala** (≈ $[0,1]$) | Amarelo: máxima preservação foveal; roxo: ≈0. |
| **`02` — $A_{ret}$** (painéis 2 e 3) | `gray` | $[0,1]$ fixa | Preto: sem atividade; branco: atividade máxima. |
| **`03` — $A_{ret}$, $A_{cortex}$, $A_{rec}$** | `gray` | **autoescala** | Ver convenção cortical abaixo. |
| **`04` — $A_{rec}$ por estágio** | `gray` | $[0,1]$ fixa | Branco: atividade preservada; preto: escotoma. |
| **`05` — Contraste e Índice de Túnel** | curvas `navy` (contraste) e `darkred` (índice) | Eixo X: $t\in\{0,\dots,5\}$; **dois painéis separados**, cada um com eixo Y próprio (autoescala) | O contraste é diferença de médias; o índice é razão de médias (≫1: forte confinamento central). |
| **`06` — grade 3×3** | `gray` | $[0,1]$ fixa | Linhas: severidade; colunas: raio preservado. |
| **`07` — campo neural 1D** | 4 curvas nas cores padrão do matplotlib | X: $x\in[-1,1]$ ($x=0$ fóvea); Y: autoescala (na figura versionada, $\approx 0$ a $6\times10^{-4}$) | Azul `linear`, laranja `alder_bifasico`, verde `michaelis_menten`, vermelho `drusas_faz`. |
| **`08` — FitzHugh-Nagumo** | `magma` | **autoescala** sobre $u$ normalizado por min–max | Cada painel é renormalizado independentemente; **cores não são comparáveis entre painéis**. |

**Convenção geométrica.** Todas as imagens usam `origin="lower"`. Nas imagens retinais o centro corresponde à fóvea; os pontos médios das bordas correspondem a $R\approx0{,}71$ e só os cantos a $R=1$. Na imagem cortical ($A_{cortex}$), o eixo horizontal é $\log R$ (fóvea à esquerda) e o vertical é o ângulo polar, que no `cv2.warpPolar` cobre $[0, 2\pi)$ (não $[-\pi,\pi]$).

### 3.2. Descrição das figuras geradas

`python main.py` gera oito arquivos de 300 dpi em `figuras/`.

1. **`01_mascara_isquemia_severidades.png`** — três painéis de $I_{ret}$ para $s = 0{,}4;\ 0{,}7;\ 0{,}95$ (raio preservado $0{,}25$, $\Delta r=0{,}05$). No caso padrão ($s=0{,}8$) o terminal imprime isquemia média de **$\approx 0{,}046$** em $R<0{,}15$ e **$0{,}800$** em $R>0{,}70$ (execução de verificação: $0{,}0455$ e $0{,}8000$), coerente com a vulnerabilidade periférica imposta por construção.

2. **`02_preservacao_foveal_e_atividade.png`** — $F_{ret}$ ($\sigma_{\text{fovea}}=0{,}12$; máximo $1{,}0$); $A_{ret}$ sem ganho ($g_{\text{fov}}=0$) e com ganho ($g_{\text{fov}}=0{,}35$). **O ganho altera pouco o centro** (média em $R<0{,}05$: $0{,}989 \to 1{,}000$) e atua sobretudo na transição ($R\in[0{,}2;0{,}3]$: $0{,}59\to 0{,}63$). A isquemia periférica já deixa o centro quase saturado.

3. **`03_mapeamento_retino_cortical.png`** — $A_{ret}$, $A_{cortex}$ e $A_{rec}$. A região central vira uma faixa clara à esquerda do mapa cortical (magnificação foveal) e a periferia ocupa a direita. A reconstrução preserva a **geometria radial qualitativa**, mas é **com perdas** (quantização, interpolação, min–max e cantos fora de `maxRadius`); não se pode afirmar ausência de distorções.

4. **`04_progressao_temporal_tunel.png`** — seis quadros ($t=0$ a $5$; $s_t$ de $0$ a $0{,}9$; $r_t$ de $0{,}55$ a $0{,}15$). **Em $t=0$ o quadro é preto**, e não "uniformemente claro": $A_{ret}\equiv1$ é uma imagem constante, que a normalização min–max transforma em zero. De $t=1$ a $t=5$ observa-se um disco claro central cada vez menor sobre fundo escuro — a aparência de "túnel". Há artefatos de interpolação do OpenCV fora do círculo inscrito (cantos), que não têm significado fisiológico.

5. **`05_metricas_contraste_tunel.png`** — as métricas **não** crescem de forma monótona nem atingem $0{,}72$ / $7{,}12$. Na figura versionada: contraste $\approx -0{,}21$ ($t=0$), $0{,}69$ ($t=1$), $\approx 0{,}79$ ($t=2$–$4$), $\approx 0{,}49$ ($t=5$); índice $0$ ($t=0$), $\approx 3{,}3$, $\approx 4{,}8$ (platô), $\approx 2{,}2$ ($t=5$). Uma reexecução independente (numpy 2.5.3, opencv-python 5.0.0.93) deu contraste $0{,}00;\ 0{,}46;\ 0{,}47;\ 0{,}50;\ 0{,}51;\ 0{,}46$ e índice $0;\ 1{,}86;\ 1{,}91;\ 2{,}01;\ 2{,}04;\ 2{,}05$. Ou seja, o comportamento é dominado pela normalização min–max, pela degeneração em $t=0$ e pelo recorte da região $R>0{,}65$; **o índice não é uma medida confiável da progressão do túnel na versão atual**, e os valores variam entre execuções.

6. **`06_varredura_parametros_3x3.png`** — severidade nas linhas ($0{,}3;\ 0{,}6;\ 0{,}9$) e raio preservado nas colunas ($0{,}15;\ 0{,}30;\ 0{,}45$), $g_{\text{fov}}=0{,}3$, $\Delta r=0{,}05$. Severidades baixas produzem apenas escurecimento difuso periférico, e severidade alta com raios pequenos produz o foco luminoso mais isolado. Isso é uma **observação visual**, não uma condição "necessária e suficiente" demonstrada.

7. **Figura 07 (arquivo `figuras/07_comparacao_campo_neural_mascaras_O2.png`)** — as quatro curvas são transientes que decaem para o ponto fixo u = 0, e não perfis estacionários: após 150 passos, o máximo de u é de aproximadamente 6 × 10⁻⁴ para todos os modos (o pico inicial era 0,6). Todas têm formato de gaussiana centrada em x = 0, e as diferenças entre modos são pequenas (o modo drusas_faz fica ligeiramente abaixo; alder_bifasico e michaelis_menten ficam ligeiramente acima e mais largos). Não aparecem ombros paracentrais de hiperexcitabilidade em |x| ≈ 0,25 nem vale em |x| ≈ 0,22. Esses efeitos são esperados pela forma de g(O), mas são muito pequenos e mascarados pelo decaimento.

8. **`08_extensao_fitzhugh_nagumo_3casos.png`** — três casos: (i) $I_{\text{cortex}}=0$; (ii) $I_{\text{cortex}}=\mathcal{R}(I_{ret})$, onde $\mathcal{R}$ é a função `retina_para_cortex`; (iii) $I_{\text{cortex}}=\mathcal{R}(\text{clip}(I_{ret}-0{,}4\,F_{ret},0,1))$, isto é, o "ganho foveal" é uma **redução de 0,4·$F_{ret}$ na isquemia**. Resultados observados: (i) $u$ é espacialmente uniforme ($u\approx-1{,}03$), logo o painel mostra apenas artefatos de normalização (min–max de campo constante), e **não** "ativação homogênea"; (ii) e (iii) são **visualmente indistinguíveis** ($u\in[-1{,}04;\,-0{,}90]$ em ambos); e a reconstrução mostra **centro escuro e anel periférico claro**, o **oposto** da hipótese de núcleo excitado circundado por silêncio. Essa extensão **não demonstra** o efeito de túnel na versão atual.

---

## 4. Discussão

Os quadros $t=1$–$5$ da Figura 04 mostram que três ingredientes são **suficientes, em uma construção fenomenológica**, para produzir uma reconstrução com foco luminoso central sobre fundo escuro: (i) perda radial periférica de atividade (imposta por $I_{ret}$), (ii) manutenção relativa da região central (a isquemia é quase nula perto do centro; o ganho foveal contribui pouco) e (iii) o mapeamento log-polar (que, na implementação, é essencialmente invertível e portanto **não é o responsável** pela aparência de túnel). Como o efeito é imposto pela máscara de isquemia, os resultados ilustram o modelo, e **não o validam** contra dados.

A comparação entre $g(O)$ (Figura 07) é apenas qualitativa: as formulações diferem na forma da função (seção 2.2), mas o campo neural decai a zero e as diferenças resultantes são mínimas. Uma análise adequada exigiria parâmetros em regime não trivial (por exemplo, maior $J$ ou $\beta$, ou entrada externa) e um critério de estado estacionário. O código **não modula** $\sigma$ de $w(y,x)$ com o contraste ou a isquemia; essa interpretação a partir de Pinotsis et al. (2014) é uma extensão futura.

**Limitações.** (a) A simetria radial desconsidera a assimetria entre disco óptico nasal e rafe temporal (Yoshimura et al., 2024) e o ângulo $\Theta$ não é usado; (b) o modelo é 2D e colapsa as três camadas capilares (SCP, ICP e DCP; Böhm et al., 2023); (c) $O\,[\%]\equiv PO_2\,[\text{mmHg}]$ não é calibrado; (d) o campo neural 1D e o FitzHugh-Nagumo 2D não estão acoplados ao pipeline retinal 2D através de $g(O)$; (e) a normalização min–max a cada etapa remove a escala absoluta e distorce métricas; (f) a região $R>0{,}65$ usada nas métricas inclui áreas fora do mapa log-polar.

---

## 5. Conclusão

O pipeline reproduz **qualitativamente** uma reconstrução em que um foco luminoso central se contrai sobre fundo escuro com o avanço da isquemia periférica (Figuras 04 e 06). As extensões dinâmicas (campo neural e FitzHugh-Nagumo) estão implementadas, mas, nos parâmetros atuais, **não** produzem os efeitos esperados (Figuras 07 e 08) e requerem recalibração. A estrutura modular (`v0.1.0` a `v1.0.0`) facilita testar patologias específicas — glaucoma de pressão normal, retinopatia diabética, DMRI — alterando a parametrização de $g(O)$ e do kernel $w(y,x)$, após corrigir os pontos listados nas seções 2.3 e 4.

---

## Referências Bibliográficas

1. **Alder, V. A., & Constable, I. J.** (1981). Effect of hypoxia on the maintained firing rate of retinal ganglion cells. *Investigative Ophthalmology & Visual Science*, 21(3), 450–456.
2. **Arciero, J., Harris, A., et al.** (2019). Capítulo sobre cinética de Michaelis-Menten em *Ocular Fluid Dynamics* (ver Guidoboni et al., 2019). *(Referência citada no código; completar dados bibliográficos.)*
3. **Böhm, E. W., Pfeiffer, N., Wagner, F. M., & Gericke, A.** (2023). Methods to measure blood flow and vascular reactivity in the retina. *Frontiers in Medicine*, 9, 1069449.
4. **Chevallier, J., Duarte, A., Löcherbach, E., & Ost, G.** (2019). Mean field limits for nonlinear spatially extended Hawkes processes with exponential memory kernels. *Stochastic Processes and their Applications*, 129(1), 1–27.
5. **Ernest, J. T., & Krill, A. E.** (1971). The effect of hypoxia on visual function: Psychophysical studies. *Investigative Ophthalmology*, 10(5), 323–328.
6. **Guidoboni, G., Harris, A., & Sacco, R. (Eds.).** (2019). *Ocular Fluid Dynamics: Anatomy, Physiology, Imaging Techniques, and Mathematical Modeling*. Birkhäuser / Springer Nature.
7. **McHugh, K. J., Li, D., Wang, J. C., Kwark, L., Loo, J., Macha, V., Farsiu, S., Kim, L. A., & Saint-Geniez, M.** (2019). Computational modeling of retinal hypoxia and photoreceptor degeneration in patients with age-related macular degeneration. *PLOS ONE*, 14(6), e0216215.
8. **Pinotsis, D. A., Brunet, N., Bastos, A., Bosman, C. A., Litvak, V., Fries, P., & Friston, K. J.** (2014). Contrast gain control and horizontal interactions in V1: A DCM study. *NeuroImage*, 92, 143–155.
9. **Yoshimura, K., Sugihara, K., Maruko, I., Iida, T., & Miura, T.** (2024). Modeling human retinal vascular pattern around foveal avascular zone. *Research Square* (Preprint), rs-4310491/v1.
