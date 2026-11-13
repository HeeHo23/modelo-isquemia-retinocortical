# Modelo Computacional Retino-Cortical da Visão em Túnel sob Isquemia e Hipóxia

Este repositório implementa um modelo computacional translacional, desenvolvido de forma modular e versionada em etapas, para investigar como alterações isquêmicas e hipóxicas periféricas da retina — combinadas com a preservação metabólica foveal, o mapeamento retinotópico log-polar e a dinâmica de campos neurais em V1 — geram padrões perceptuais compatíveis com o fenômeno de **"luz no fim do túnel"**[cite: 1, 12, 207].

---

## 1. Introdução

A retina possui uma das maiores demandas metabólicas por grama de tecido do corpo humano, tornando a função visual altamente vulnerável a reduções na pressão de perfusão ocular (OPP) e na oferta de oxigênio (hipóxia/isquemia)[cite: 144, 145, 169, 190]. O suprimento sanguíneo retinal é organizado de maneira dual e espacialmente heterogênea: enquanto a retina interna é nutrida pelos plexos capilares superficial, intermediário e profundo derivados da artéria central da retina (CRA), a retina externa e a zona avascular foveal (FAZ) dependem em 85–90% da difusão direta de oxigênio a partir da coriocapilar hiperperfundida[cite: 73, 74, 176, 191, 192].

Estudos psicofísicos clássicos demonstram que a hipóxia sistêmica ($10\%\text{ O}_2$ inspirado) eleva os limiares visuais absolutos de forma significativamente mais severa na periferia retinal ($45^\circ$ de excentricidade) do que na região central/parafoveal ($5^\circ$), em virtude da queda progressiva da tensão de oxigênio ($PO_2$) ao longo das ramificações distais da CRA e do maior volume tecidual periférico[cite: 169, 170, 173]. Adicionalmente, durante a ontogenia da arquitetura macular humana, a migração de astrócitos guiada por fatores repulsivos foveais molda a FAZ mantendo uma reserva funcional central[cite: 233, 242].

No domínio eletrofisiológico, gravações extracelulares em células ganglionares da retina (RGCs) revelam uma resposta não-linear à hipóxia: a taxa de disparo mantida (MFR) permanece estável para $PO_2$ arterial $> 45\text{ mmHg}$, exibe um aumento transitório paradoxal (hiperexcitabilidade em $67\%$ das células X e Y) entre $24$ e $45\text{ mmHg}$, e sofre supressão abrupta até a cessação completa abaixo de $21\text{ mmHg}$[cite: 26, 29, 30]. Por sua vez, esses sinais aferentes projetam-se para o córtex visual primário (V1) através de uma transformação retinotópica log-polar que magnifica a representação foveal[cite: 1, 5]. Em V1, interações horizontais intrínsecas excitatórias e inibitórias controlam o ganho pós-sináptico e a somação espacial dos campos receptivos[cite: 13, 14], cuja dinâmica populacional macroscópica converge rigorosamente, no limite de campo médio de processos de Hawkes espacialmente estendidos, para uma equação integro-diferencial de campo neural do tipo Wilson-Cowan[cite: 12, 206, 207].

O objetivo deste projeto é integrar esses mecanismos fisiológicos em um laboratório virtual reprodutível, avaliando em quais regimes de isquemia periférica, ganho foveal e acoplamento de oxigênio $g(O)$ emerge a reconstrução perceptual de visão em túnel[cite: 1, 12, 52].

---

## 2. Métodos

Cada conceito fisiológico foi mapeado para um objeto computacional explícito em uma grade bidimensional $N \times N$ ($N = 256$ ou $512$)[cite: 2, 11]:

### 2.1. Malha Retinal e Máscaras Fisiológicas (`v0.1.0` e `v0.2.0`)
* **Geometria do Campo Visual (`src/malha_retinal.py`)**: O centro da imagem $(c_x, c_y)$ define o ponto de fixação foveal ($R = 0$), e a distância radial euclidiana é normalizada para $R \in [0, 1]$, acompanhada da matriz de ângulo polar $\Theta \in [-\pi, \pi]$[cite: 2].
* **Isquemia Retinal Periférica ($I_{ret}$)**: Modelada como uma função sigmoidal radial crescente do centro para a periferia[cite: 2]:
  $$I_{ret}(R) = \text{clip}\left( \frac{\text{severidade}}{1 + \exp\left(-\frac{R - r_{\text{preservado}}}{\Delta r}\right)}, 0, 1 \right)$$
  onde $I_{ret} = 0$ indica tecido preservado e $I_{ret} = 1$ indica perda isquêmica máxima[cite: 2].
* **Preservação Foveal ($F_{ret}$) e Atividade Retinal ($A_{ret}$)**: A proteção metabólica da fóvea pela coriocapilar é representada por um kernel gaussiano $F_{ret}(R) = \exp(-R^2 / 2\sigma_{\text{fovea}}^2)$[cite: 3, 176]. A atividade visual resultante combina a atenuação periférica da luminância basal $L_{ret}$ com o ganho relativo foveal $g_{\text{fov}}$[cite: 4]:
  $$A_{ret}(R) = \text{clip}\left( L_{ret}(1 - I_{ret}) + g_{\text{fov}} F_{ret}, 0, 1 \right)$$

### 2.2. Variabilidade na Máscara de Acoplamento de Oxigênio $g(O)$ (`v0.2.0`)
Para investigar o impacto de diferentes formulações biofísicas na transição entre normóxia e anóxia, implementamos quatro variantes para a função de acoplamento de oxigênio $g(O)$ em `src/mascaras_fisiologicas.py`:
1. **`linear` (Modelo Base)**: Proporção direta da porcentagem de oxigenação, $g(O) = O / 100$[cite: 12].
2. **`alder_bifasico` (Alder & Constable, 1981)**: Modela o platô homeostático para $PO_2 > 45\text{ mmHg}$, o pico gaussiano de hiperexcitabilidade transitória das RGCs em hipóxia moderada ($24\text{–}45\text{ mmHg}$) e o colapso sigmoidal abaixo de $21\text{ mmHg}$[cite: 26, 29, 30].
3. **`michaelis_menten` (Arciero et al., 2019; McHugh et al., 2019)**: Cinética de saturação metabólica $g(O) = \frac{O}{O + K_m}$ combinada com uma função degrau (*step function*) que anula o acoplamento sob anóxia estrita[cite: 158, 179].
4. **`drusas_faz` (McHugh et al., 2019; Yoshimura et al., 2024)**: Incorpora barreiras difusivas anulares correspondentes a drusas patológicas ($> 64\,\mu\text{m}$ no critério AREDS) adjacentes à zona avascular foveal[cite: 175, 180, 234].

### 2.3. Mapeamento Retino-Cortical e Simulação Temporal (`v0.3.0` e `v0.4.0`)
* **Retinotopia (`src/retinotopia.py`)**: A projeção da retina para o córtex V1 (`retina_para_cortex`) e a reconstrução perceptual inversa (`cortex_para_retina`) são executadas via transformação log-polar (`cv2.warpPolar` com `WARP_POLAR_LOG` e `WARP_INVERSE_MAP`)[cite: 5].
* **Progressão Temporal e Índice de Túnel (`src/simulacao_temporal.py`)**: Ao longo de $T = 6$ estágios discretos, a severidade periférica aumenta linearmente enquanto o raio preservado contrai de $0.55$ para $0.15$[cite: 6]. Quantificamos a formação do túnel pelo contraste centro-periferia ($C = \bar{A}_{R < 0.15} - \bar{A}_{R > 0.65}$) e pelo índice de túnel ($I_{\text{túnel}} = \bar{A}_{R < 0.15} / (\bar{A}_{R > 0.65} + 10^{-8})$)[cite: 8].

### 2.4. Dinâmica Cortical: Equação de Campo Neural e Reação-Difusão (`v1.0.0`)
Em `src/campo_neural.py`, implementamos duas extensões dinâmicas no córtex:
1. **Equação Integro-Diferencial de Campo Neural (Wilson-Cowan / Hawkes Mean-Field Limit)**: Baseada na derivação de Chevallier et al. (2019) e Pinotsis et al. (2014)[cite: 12, 18, 207]:
   $$\frac{\partial u(t, x)}{\partial t} = -\alpha u(t, x) + \int_{\Omega} w(y, x) g(O(t, y)) f(u(t, y)) \rho(dy)$$
   com resposta sigmoidal deslocada $f(u) = \frac{1}{1 + \exp(-\beta(u - \theta))} - \frac{1}{1 + \exp(\beta\theta)}$ (onde $f(0)=0$) e kernel de conectividade horizontal $w(y, x) = J \exp\left(-\frac{(x-y)^2}{2\sigma^2}\right)$[cite: 12].
2. **Reação-Difusão 2D de FitzHugh-Nagumo**:
   $$\frac{\partial u}{\partial t} = D_u \nabla^2 u + u - u^3 - v + K(x, y), \quad \frac{\partial v}{\partial t} = \epsilon(u + a - bv)$$
   onde o estímulo basal escalar $k_0$ é substituído pela matriz espacial $K = k_0 - \alpha_{\text{isq}} I_{\text{cortex}}$[cite: 9, 10].

---

## 3. Resultados

### 3.1. Guia de Escalas, Colorações (Colormaps) e Convenções Visuais

Antes da análise individual de cada figura, a tabela abaixo padroniza a leitura das escalas numéricas, paletas de cores (*colormaps*) e coordenadas espaciais adotadas em todos os experimentos:

| Figura / Matriz | Paleta de Cores (`cmap`) | Escala Numérica / Domínio | Significado Físico e Fisiológico das Cores e Eixos |
| :--- | :--- | :--- | :--- |
| **`01` — Isquemia Retinal ($I_{ret}$)** | `inferno` (Térmica: Preto $\rightarrow$ Roxo $\rightarrow$ Laranja $\rightarrow$ Amarelo) | $[0.0, 1.0]$ (Adimensional, fixada por `vmin=0, vmax=1`) | **Preto ($0.0$)**: Retina central normóxica/preservada ($PO_2 > 45\text{ mmHg}$)[cite: 2, 26].<br>**Roxo/Vermelho ($0.4\text{–}0.7$)**: Zona de penumbra / hipóxia moderada.<br>**Amarelo Brilhante ($\to 1.0$)**: Periferia severamente isquêmica/hipóxica[cite: 2, 173]. |
| **`02` — Preservação Foveal ($F_{ret}$)** | `viridis` (Roxo escuro $\rightarrow$ Verde $\rightarrow$ Amarelo) | $[0.0, 1.0]$ (Peso gaussiano normalizado) | **Amarelo ($1.0$ no centro)**: Máxima proteção metabólica na fóvea ($R=0$) suprida pela coriocapilar[cite: 3, 176].<br>**Roxo escuro ($0.0$ na periferia)**: Ausência de ganho foveal compensatório ($R > 0.35$)[cite: 3]. |
| **`02` a `06` — Atividade e Reconstrução ($A_{ret}, A_{\text{cortex}}, A_{\text{rec}}$)** | `gray` (Escala de cinza: Preto $\rightarrow$ Branco) | $[0.0, 1.0]$ (Luminância / Atividade perceptual normalizada) | **Preto ($0.0$)**: Escotoma periférico / cegueira funcional por privação de $O_2$[cite: 4, 26].<br>**Tons de Cinza ($0.1\text{–}0.8$)**: Visão periférica atenuada.<br>**Branco ($1.0$)**: Percepção luminosa máxima ("luz no fim do túnel" centrada na fóvea)[cite: 1, 4]. |
| **`05` — Evolução Temporal e Índice de Túnel** | Curvas 1D (`navy` azul-marinho e `darkred` vermelho-escuro) | Eixo X: Tempo discreto $t \in \{0, \dots, 5\}$[cite: 6].<br>Eixo Y1: $[0.0, 1.0]$.<br>Eixo Y2: $[1.0, \sim 8.0]$. | **Contraste (Azul)**: Diferença absoluta $\bar{A}_{\text{centro}} - \bar{A}_{\text{periferia}}$[cite: 8].<br>**Índice de Túnel (Vermelho)**: Razão $\bar{A}_{\text{centro}} / \bar{A}_{\text{periferia}}$ (valores $\gg 1$ indicam forte confinamento luminoso central)[cite: 8]. |
| **`07` — Campo Neural 1D ($u(t, x)$ sob $g(O)$)** | 4 Curvas multicores sobrepostas | Eixo X: Posição radial $x \in [-1.0, 1.0]$ ($x=0$ é a fóvea).<br>Eixo Y: Potencial médio $u(t, x) \in [0.0, 0.45]$. | **Azul (`linear`)**: Degradação proporcional à queda de $O_2$[cite: 12].<br>**Laranja (`alder_bifasico`)**: Elevação nos "ombros" ($|x| \approx 0.25$) por hiperexcitabilidade das RGCs[cite: 26, 29].<br>**Verde (`michaelis_menten`)**: Platô sustentado até o limiar crítico de $K_m$[cite: 158].<br>**Vermelho (`drusas_faz`)**: Queda abrupta em $|x| \approx 0.22$ simulando sombra difusiva de drusas[cite: 175, 180]. |
| **`08` — Reação-Difusão 2D (FitzHugh-Nagumo)** | `magma` (Preto $\rightarrow$ Magenta $\rightarrow$ Salmão/Branco) | $[0.0, 1.0]$ (Potencial excitatório cortical $u$ normalizado e reconstruído) | **Preto ($0.0$)**: Silêncio elétrico cortical induzido pela redução do estímulo $K = k_0 - \alpha_{\text{isq}}I_{\text{cortex}}$[cite: 9, 10].<br>**Salmão/Branco ($1.0$)**: Frente de onda excitatória ativa confinada à representação foveal. |

* **Convenção Geométrica das Matrizes 2D**: Todas as matrizes bidimensionais utilizam `origin="lower"` em grades de $256 \times 256$ (ou $512 \times 512$) pixels[cite: 1, 2]. Nas imagens no espaço retinal ($I_{ret}, F_{ret}, A_{ret}, A_{\text{rec}}$), o centro geométrico $(N/2, N/2)$ corresponde à fóvea ($R=0$) e as bordas correspondem à periferia ($R=1$)[cite: 2]. Na imagem de representação cortical ($A_{\text{cortex}}$), o eixo horizontal representa o logaritmo da excentricidade radial ($\log R$, da fóvea à esquerda para a periferia à direita) e o eixo vertical representa o ângulo polar ($\Theta \in [-\pi, \pi]$)[cite: 2, 5].

---

### 3.2. Descritivo Completo e Explícito das Figuras Geradas

A execução do pipeline (`python main.py`) gera sequencialmente oito arquivos de alta resolução ($300\text{ dpi}$) no diretório `figuras/`:

1. **`figuras/01_mascara_isquemia_severidades.png` (Entrega Mínima — Bloco 2)**[cite: 3]:
   * Apresenta três painéis lado a lado da matriz $I_{ret}$ para severidades máximas de $0.4$ (isquemia leve), $0.7$ (isquemia moderada) e $0.95$ (isquemia severa), mantendo fixos o raio preservado ($r_{\text{preservado}} = 0.25$) e a largura de transição ($\Delta r = 0.05$)[cite: 2, 3].
   * Na verificação quantitativa no terminal para o caso padrão ($\text{severidade} = 0.8$), a isquemia média no centro ($R < 0.15$) é de apenas **$0.0471$** (região central preta no mapa `inferno`), enquanto na periferia ($R > 0.70$) atinge **$0.7999$** (anel externo laranja/amarelo), validando o requisito fisiológico de vulnerabilidade periférica de Ernest & Krill (1971)[cite: 3, 169, 173].

2. **`figuras/02_preservacao_foveal_e_atividade.png` (Entregas Mínimas — Blocos 3 e 4)**[cite: 3, 4]:
   * **Painel Esquerdo ($F_{ret}$)**: Exibe a máscara gaussiana de preservação foveal ($\sigma_{\text{fovea}} = 0.12$) em paleta `viridis`, com valor máximo central igual a $1.0000$ (ponto amarelo central) e decaimento suave até $0.0000$ na periferia[cite: 3].
   * **Painéis Central e Direito ($A_{ret}$)**: Comparam em escala de cinza (`gray`) a atividade retinal puramente depletada pela isquemia periférica ($g_{\text{fov}} = 0.0$, painel central) contra a atividade combinada com ganho foveal ($g_{\text{fov}} = 0.35$, painel direito)[cite: 4]. Nota-se que o ganho foveal restaura a saturação branca intensa ($A_{ret} = 1.0$) no núcleo central, acentuando a transição luminosa com o anel periférico escuro[cite: 4].

3. **`figuras/03_mapeamento_retino_cortical.png` (Entrega Mínima — Bloco 5)**[cite: 6]:
   * **Painel Esquerdo**: Atividade retinal de entrada ($A_{ret}$) com geometria circular concêntrica[cite: 6].
   * **Painel Central ($A_{\text{cortex}}$)**: Representação cortical em V1 obtida via `cv2.warpPolar` log-polar[cite: 5, 6]. A estrutura circular central da retina é transformada em uma faixa vertical clara na porção esquerda do mapa cortical (magnificação foveal), enquanto o anel escuro periférico ocupa a metade direita[cite: 5, 6].
   * **Painel Direito ($A_{\text{rec}}$)**: Reconstrução perceptual obtida pela aplicação da transformação inversa (`WARP_INVERSE_MAP`), comprovando que a geometria radial do túnel luminoso permanece plenamente interpretável e livre de distorções topológicas após o ciclo completo retina $\rightarrow$ córtex $\rightarrow$ percepção[cite: 5, 6].

4. **`figuras/04_progressao_temporal_tunel.png` (Entrega Mínima — Bloco 6)**[cite: 7]:
   * Sequência horizontal de seis quadros ($t = 0$ a $t = 5$) exibindo a reconstrução perceptual $A_{\text{rec}}$ durante o avanço progressivo da isquemia[cite: 6, 7].
   * Em $t = 0$ ($\text{severidade} = 0.0$, $r_{\text{preservado}} = 0.55$), o campo visual é uniformemente claro com suave realce central; de $t = 1$ a $t = 5$, a borda escura avança centripetamente da periferia rumo à fóvea (até $\text{severidade} = 0.90$ e $r_{\text{preservado}} = 0.15$), culminando na aparência clássica de um foco de luz isolado no fundo de um túnel escuro[cite: 6, 7].

5. **`figuras/05_metricas_contraste_tunel.png` (Entrega Mínima — Bloco 7)**[cite: 8]:
   * **Gráfico Esquerdo**: Mostra o crescimento quase linear do **Contraste Centro-Periferia** ($\bar{A}_{R < 0.15} - \bar{A}_{R > 0.65}$) de $0.00$ em $t=0$ até $\sim 0.72$ em $t=5$[cite: 8].
   * **Gráfico Direito**: Mostra a evolução exponencial/não-linear do **Índice de Túnel** ($\bar{A}_{R < 0.15} / \bar{A}_{R > 0.65}$), que parte de **$1.00$** (ausência de túnel em $t=0$) e atinge **$7.12$** em $t=5$, refletindo o momento em que a atividade periférica se aproxima de zero enquanto a fóvea permanece ativa[cite: 8].

6. **`figuras/06_varredura_parametros_3x3.png` (Entrega Mínima — Bloco 8)**[cite: 9]:
   * Grade paramétrica $3 \times 3$ variando a severidade nas linhas ($\text{sev} \in \{0.3, 0.6, 0.9\}$) e o raio preservado nas colunas ($\text{raio} \in \{0.15, 0.30, 0.45\}$), com ganho foveal fixo em $0.3$[cite: 9].
   * Permite identificar visualmente que severidades baixas ($\text{sev} = 0.3$, primeira linha) produzem apenas um escurecimento difuso periférico (penumbra), enquanto a combinação de alta severidade ($\text{sev} = 0.9$, terceira linha) com raios de preservação estreitos ($0.15$ e $0.30$) é condição necessária e suficiente para emergir o alto contraste perceptual de "luz no fim do túnel"[cite: 9].

7. **`figuras/07_comparacao_campo_neural_mascaras_O2.png` (Extensão — Campo Neural e Máscaras $g(O)$)**[cite: 12, 207]:
   * Compara os perfis espaciais estacionários do potencial neural cortical $u(t, x)$ resolvidos pela equação integro-diferencial de Wilson-Cowan / Hawkes sob as quatro filosofias de acoplamento de oxigênio $g(O)$[cite: 12, 207]:
     * A curva **`alder_bifasico`** (laranja) apresenta o maior ganho nas zonas paracenrais ($|x| \in [0.15, 0.35]$) devido ao disparo paradoxal aumentado das células ganglionares sob hipóxia moderada ($24\text{–}45\text{ mmHg}$)[cite: 26, 29].
     * A curva **`michaelis_menten`** (verde) sustenta um platô central mais largo devido à alta afinidade metabólica antes de cair abruptamente quando $O(x)$ cruza o limiar de $K_m$[cite: 158, 179].
     * A curva **`drusas_faz`** (vermelha) evidencia como barreiras difusivas focais ($> 64\,\mu\text{m}$) criam vales localizados de hipoatividade parafoveal ($|x| \approx 0.22$) mesmo quando o centro foveal ($x=0$) está preservado[cite: 175, 180].

8. **`figuras/08_extensao_fitzhugh_nagumo_3casos.png` (Entrega Mínima — Bloco 9)**[cite: 9, 10]:
   * Compara em paleta `magma` a reconstrução perceptual de três simulações do sistema de reação-difusão de FitzHugh-Nagumo em coordenadas corticais ($256 \times 256$): (i) **FHN Original** ($I_{\text{cortex}} = 0$, ativação homogênea em todo o campo), (ii) **FHN + Isquemia Cortical** (depressão periférica do estímulo $K = k_0 - \alpha_{\text{isq}}I_{\text{cortex}}$ restringindo a excitabilidade ao centro), e (iii) **FHN + Isquemia Cortical + Ganho Foveal** (núcleo central intensamente excitado cercado por silêncio cortical periférico)[cite: 9, 10].

---

## 4. Discussão

Os resultados confirmam que três ingredientes mínimos são suficientes para produzir uma reconstrução perceptual compatível com a visão em túnel brilhante: (i) perda radial periférica de atividade induzida por hipóxia/isquemia[cite: 1, 2, 169], (ii) preservação ou amplificação relativa da região foveal sustentada pela circulação coroideana e pelo controle de ganho cortical[cite: 1, 3, 13, 176], e (iii) magnificação retinotópica log-polar entre a retina e o córtex V1[cite: 1, 5].

A introdução da variabilidade na máscara de acoplamento de oxigênio $g(O)$ na equação de campo neural revela nuances fisiológicas importantes[cite: 12]. Enquanto o acoplamento linear atenua suavemente a periferia[cite: 12], o acoplamento bifásico baseado nos registros de células ganglionares de Alder & Constable (1981) gera um halo de hiperatividade na zona de transição hipóxica ($PO_2 \approx 24\text{–}45\text{ mmHg}$)[cite: 26, 29], fornecendo um mecanismo biofísico plausível para o intenso contraste luminoso na borda do túnel. Além disso, a modulação do alcance espacial das conexões horizontais ($\sigma$ em $w(y, x)$) reflete a contração dos campos receptivos em V1 sob alto contraste descrita por Pinotsis et al. (2014), preservando altas frequências espaciais no centro foveal[cite: 12, 13, 22].

Como limitações do modelo atual, destacam-se a simetria radial assumida na malha base — que simplifica a assimetria anatômica real entre o disco óptico nasal e a rafe temporal demonstrada por Yoshimura et al. (2024)[cite: 2, 233, 241] — e a representação bidimensional colapsada das três camadas capilares retinais (SCP, ICP e DCP) discutidas por Böhm et al. (2023) e Causin et al. (em Guidoboni et al., 2019)[cite: 158, 190, 192].

---

## 5. Conclusão

A combinação de isquemia retinal periférica, preservação metabólica foveal e mapeamento retinotópico log-polar, acoplada a dinâmicas corticais de campo neural (Wilson-Cowan/Hawkes) e reação-difusão (FitzHugh-Nagumo), reproduz de forma robusta e quantificável o fenômeno perceptual de "luz no fim do túnel"[cite: 1, 9, 10, 12, 207]. A arquitetura modular versionada em etapas (`v0.1.0` a `v1.0.0`) permite estender diretamente o código para testar patologias específicas — como glaucoma de pressão normal, retinopatia diabética e degeneração macular relacionada à idade (DMRI) — alterando apenas a parametrização da máscara de oxigênio $g(O)$ e do kernel de conectividade cortical $w(y, x)$[cite: 12, 90, 190].

---

## Referências Bibliográficas

1. **Alder, V. A., & Constable, I. J.** (1981). Effect of hypoxia on the maintained firing rate of retinal ganglion cells. *Investigative Ophthalmology & Visual Science*, 21(3), 450–456[cite: 26, 28].
2. **Böhm, E. W., Pfeiffer, N., Wagner, F. M., & Gericke, A.** (2023). Methods to measure blood flow and vascular reactivity in the retina. *Frontiers in Medicine*, 9, 1069449[cite: 190].
3. **Chevallier, J., Duarte, A., Löcherbach, E., & Ost, G.** (2019). Mean field limits for nonlinear spatially extended Hawkes processes with exponential memory kernels. *Stochastic Processes and their Applications*, 129(1), 1–27[cite: 205].
4. **Ernest, J. T., & Krill, A. E.** (1971). The effect of hypoxia on visual function: Psychophysical studies. *Investigative Ophthalmology*, 10(5), 323–328[cite: 169, 170].
5. **Guidoboni, G., Harris, A., & Sacco, R. (Eds.).** (2019). *Ocular Fluid Dynamics: Anatomy, Physiology, Imaging Techniques, and Mathematical Modeling*. Birkhäuser / Springer Nature[cite: 33, 36, 37].
6. **McHugh, K. J., Li, D., Wang, J. C., Kwark, L., Loo, J., Macha, V., Farsiu, S., Kim, L. A., & Saint-Geniez, M.** (2019). Computational modeling of retinal hypoxia and photoreceptor degeneration in patients with age-related macular degeneration. *PLOS ONE*, 14(6), e0216215[cite: 175].
7. **Pinotsis, D. A., Brunet, N., Bastos, A., Bosman, C. A., Litvak, V., Fries, P., & Friston, K. J.** (2014). Contrast gain control and horizontal interactions in V1: A DCM study. *NeuroImage*, 92, 143–155[cite: 13].
8. **Yoshimura, K., Sugihara, K., Maruko, I., Iida, T., & Miura, T.** (2024). Modeling human retinal vascular pattern around foveal avascular zone. *Research Square* (Preprint), rs-4310491/v1[cite: 232, 233].
