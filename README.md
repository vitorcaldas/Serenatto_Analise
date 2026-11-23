# Serenatto_Analise

# Projeto Serenatto — Diagnóstico de Vendas, Canais e Satisfação de Clientes

Análise exploratória, inferência estatística e modelagem preditiva aplicada aos dados operacionais e de experiência do cliente da rede de cafeterias **Serenatto**.

## 🎯 Objetivo do Projeto
Diagnosticar o comportamento de receita, a frequência de consumo, o desempenho comparativo entre canais de venda e os fatores associados à satisfação dos clientes, fornecendo recomendações estratégicas baseadas em evidências auditadas e estatisticamente robustas.

## 2. Cenário de Negócio Analisado

A **Serenatto** é uma operação de cafeterias com atuação multicanal (**Loja física, App e Delivery**) e presença regional (São Paulo, Curitiba e Belo Horizonte)[cite: 1, 3]. A diretoria recebeu um relatório preliminar de uma equipe anterior sugerindo que a operação estava "saudável e excelente", recomendando **manter a estratégia atual e investir na expansão para novas cidades e na experiência presencial**[cite: 1].

O papel da auditoria foi verificar se os dados, testes estatísticos, modelos de Machine Learning e gráficos sustentavam uma decisão de alto impacto financeiro (expansão geográfica e alocação de CAPEX em lojas físicas)[cite: 1, 2, 3, 4]. O diagnóstico revelou que a narrativa otimista era fruto de **erros metodológicos acumulados** — incluindo perda de dados na limpeza, confusão entre item e pedido, amostras enviesadas, vazamento de dados na avaliação do modelo e gráficos com escalas distorcidas[cite: 1, 2, 3, 4].

---

## 3. Decisões Técnicas e Metodológicas Tomadas

1. **Preservação de Dados e Padronização Categórica (Etapa 1):**
   * Substituição do descarte indiscriminado (`df.dropna()`, que eliminava ~5,1% da base) por tratamento seletivo por escopo: registros sem nota de satisfação permanecem válidos para análises de receita, enquanto campos monetários passam por higienização robusta e auditoria de consistência (`preco_unitario * quantidade == valor_total`)[cite: 3].
   * Normalização textual (`.str.strip()`, padronização de caixa e mapeamento canônico) em `cidade` e `canal_venda` para evitar fragmentação de métricas (ex.: `"São Paulo"` vs. `"são paulo"`, `"Loja"` vs. `"loja"`)[cite: 3].
2. **Reenquadramento Estatístico para Distribuições Assimétricas (Etapa 1):**
   * Adoção de **mediana (R$ 24,00) e IQR (R$ 12,00 – R$ 45,00)** ao lado da média (R$ 32,34), explicitando que o valor refere-se ao **registro/item** e não ao pedido completo[cite: 1, 3].
3. **Bloqueio de Extrapolação Amostral e Alinhamento de Hipóteses (Etapa 2):**
   * Rejeição do intervalo de confiança $[3,40; 4,20]$ visitas/semana obtido na fila da unidade SP — Centro como métrica da marca, devido ao *length-biased sampling* (clientes muito frequentes têm probabilidade proporcionalmente maior de serem abordados na fila)[cite: 4].
   * Correção do teste comparativo entre **Loja e App**: alinhamento entre a hipótese direcional ($H_1: \mu_{\text{Loja}} > \mu_{\text{App}}$) e o teste **t de Welch unilateral**, acompanhado do cálculo de tamanho de efeito ($d$ de Cohen $\approx 0,12$) e testes não paramétricos (Mann-Whitney U) para lidar com a assimetria[cite: 4].
4. **Correção de Data Leakage e Métricas de Negócio no Modelo (Etapa 3):**
   * Substituição da avaliação feita sobre `X_train` pela avaliação estrita em **`X_test` com divisão estratificada (`stratify=y`)**[cite: 2].
   * Inclusão de variáveis categóricas/operacionais (`canal_venda`, `cidade`, `categoria`, sazonalidade) via `Pipeline` + `ColumnTransformer`, eliminando a colinearidade redundante entre `preco_unitario`, `quantidade` e `valor_total`[cite: 2].
   * Troca da acurácia global (86%, apenas 2 p.p. acima de um classificador ingênuo que chuta sempre a classe majoritária de 84%) por **Recall e F1-score da classe 0 (insatisfação)** e **PR-AUC**[cite: 2].
5. **Integridade Visual na Comunicação Executiva (Etapas 3 e 4):**
   * Substituição do gráfico de pizza de 12 fatias por **barras horizontais ordenadas** com rótulos diretos (análise de Pareto)[cite: 2].
   * Correção do gráfico de receita mensal para iniciar o **eixo Y em R$ 0** (eliminando o corte artificial em R$ 9.500 que ampliava visualmente variações de 20% para 500%), remoção de cores redundantes e adoção de títulos factuais[cite: 2].

---

## 4. Ferramentas e Frameworks para Estruturar o Raciocínio

* **Auditoria de Linhagem e Rastreabilidade de Artefatos:** Mapeamento ponta a ponta entre cada afirmação executiva do e-mail à diretoria e sua origem técnica (código, tabela, teste ou gráfico), classificando falhas por gravidade de impacto no negócio[cite: 1].
* **Teoria da Amostragem e Inferência Estatística:** Avaliação de vieses de seleção e frequência, adequação paramétrica vs. não paramétrica e separação entre **significância estatística** ($p$-valor) e **relevância prática** (tamanho de efeito e margem financeira)[cite: 4].
* **Engenharia de Pipelines e Validação de ML (`scikit-learn`):** Prevenção de vazamento de dados (*data leakage*), uso de baselines ingênuos (*Dummy Classifier* / regra da classe majoritária) como régua mínima de performance e análise de matriz de confusão orientada ao custo de erro[cite: 2].
* **Princípios de Data Storytelling e Percepção Visual (Gestalt / Tufte / Cole Nussbaumer Knaflic):** Redução de carga cognitiva, proporcionalidade gráfica (regra do eixo zero em barras) e foco pré-atentivo[cite: 2].

---

## 5. Trade-offs Avaliados

| Dimensão | Opção A (Abordagem Anterior / Ingênua) | Opção B (Abordagem Auditada / Adotada) | Trade-off e Justificativa |
| :--- | :--- | :--- | :--- |
| **Tratamento de Nulos** | `dropna()` global (rápido, porém descarta 247 vendas válidas)[cite: 3]. | Tratamento condicional por análise (mantém receita sem nota; filtra apenas para modelagem de satisfação)[cite: 3]. | Aumenta ligeiramente a complexidade do código, mas preserva ~5,1% da amostra financeira e evita viés de seleção[cite: 3]. |
| **Métrica do Modelo** | Otimizar Acurácia geral (86%)[cite: 2]. | Priorizar **Recall da Classe 0** (insatisfação), mesmo reduzindo a acurácia global[cite: 2]. | Aceita-se gerar alguns falsos positivos (alertas preventivos de atendimento) para evitar falsos negativos críticos (deixar escapar 80% dos clientes insatisfeitos)[cite: 2]. |
| **Velocidade vs. Rigor na Coleta** | Pesquisa rápida na fila ($n=35$, baixo custo e entrega imediata)[cite: 4]. | Amostragem aleatória estratificada por cadastro/canais ($n \approx 300\text{--}400$)[cite: 4]. | Posterga a conclusão sobre fidelidade, mas evita alocar milhões em expansão com base em uma ilusão estatística[cite: 1, 4]. |
| **Comunicação à Diretoria** | Mensagem assertiva e otimista ("expandir agora")[cite: 1]. | Mensagem cautelosa com ressalvas metodológicas e plano de saneamento[cite: 1]. | Reduz o entusiasmo imediato no curto prazo, mas protege a credibilidade da área de Dados e mitiga o risco de prejuízo operacional[cite: 1]. |

---

## 6. Principais Recomendações Geradas

1. **Segurar a Expansão Geográfica Imediata (Decisão Estratégica):** Não aprovar abertura de unidades em novas cidades até integrar dados de **custos, margens líquidas e demanda regional**, confirmando a rentabilidade real por pedido completo[cite: 1].
2. **Revisar a Alocação de Investimentos entre Loja e App (Decisão Comercial):** Não direcionar investimentos exclusivamente para lojas físicas com base apenas na diferença bruta de R$ 3,30 no ticket[cite: 1, 4]. Avaliar taxas de plataforma, custo de ocupação, composição do carrinho (compras em grupo vs. individuais) e margem líquida por canal[cite: 1, 4].
3. **Investigar Bolsões de Insatisfação (Decisão Operacional):** Apesar da mediana de satisfação ser 4, há registros de nota mínima (1)[cite: 1, 3]. Cruzar notas baixas por `canal_venda`, `cidade` e `produto` (ex.: tempo de entrega no Delivery ou rupturas na Loja) para ações corretivas focalizadas[cite: 1, 3].
4. **Reestruturar a Coleta de Fidelidade (Decisão de Pesquisa):** Substituir entrevistas em fila por acompanhamento longitudinal via ID de cliente/pedido no App e amostragem estratificada em todas as praças e turnos[cite: 4].
5. **Bloquear o Deploy do Modelo Preditivo Atual (Decisão de Engenharia de Dados):** Manter o modelo fora de produção até que seja validado em conjunto de teste isolado, incorpore variáveis operacionais reais (tempo de preparo/entrega, canal, loja) e atinja sensibilidade aceitável na detecção de clientes detratores[cite: 1, 2].
