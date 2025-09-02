# 🩺 Projeto CardioIA: Base de Dados para Inteligência Artificial em Cardiologia

Bem-vindo ao repositório do Projeto CardioIA. Este projeto representa a fase fundamental de **coleta, organização e preparação de uma base de dados multimodal** (numérica, textual e visual) para o desenvolvimento de futuras soluções de Inteligência Artificial aplicadas à saúde cardiovascular.

Este documento serve como guia central, explicando os objetivos, as fontes de dados e a estrutura de cada componente do projeto.

## 🎯 Objetivos do Projeto

* **Coletar Dados Multimodais:** Reunir dados numéricos, textuais e visuais de fontes públicas e confiáveis para criar um repositório rico e diversificado.
* **Organizar e Limpar:** Estruturar os dados de forma lógica e aplicar técnicas de limpeza para garantir a qualidade e a consistência, com foco em governança de dados.
* **Análise Exploratória (EDA):** Realizar uma análise inicial para extrair os primeiros insights e compreender as características de cada conjunto de dados.
* **Construir uma Base Sólida:** Criar um alicerce de dados preparado para as próximas fases de um projeto de IA, como treinamento de modelos preditivos, sistemas de NLP e algoritmos de visão computacional.

---

## 📊 Parte 1 – Dados Numéricos (Análise Preditiva)

Esta seção foca em dados estruturados, tipicamente provenientes de exames clínicos e medições de pacientes, que poderiam ser coletados por dispositivos (IoT) ou inseridos em prontuários eletrônicos.

* **Descrição:** O conjunto de dados contém 14 atributos clínicos de pacientes, como idade, sexo, nível de colesterol, pressão arterial e resultados de testes de esforço.
* **Objetivo Específico:** Preparar os dados para treinar um modelo de Machine Learning capaz de **prever a presença ou ausência de doença cardíaca** com base nesses atributos.
* **Fonte Principal:**
    * **Dataset:** Heart Disease UCI
    * **Link:** [Kaggle - Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
* **Ferramentas Utilizadas:** Python, Pandas, Matplotlib, Seaborn.

Análise do dataset UCI Heart Disease Data

### 1. A 🌍 Origem dos Dados: Reais e com História

Este é um dos datasets mais clássicos e respeitados na comunidade de Machine Learning para cardiologia. Ele é conhecido como "Heart Disease Dataset" do repositório da UCI (University of California, Irvine).

Coleta Real: Os dados foram coletados de pacientes reais em quatro hospitais diferentes ao redor do mundo: a famosa Cleveland Clinic (EUA), o Instituto Húngaro de Cardiologia, o Hospital Universitário de Zurique (Suíça) e o V.A. Medical Center em Long Beach (EUA). Cada linha representa a jornada de um paciente real.

Propósito Científico: Foram coletados por médicos e pesquisadores com o objetivo de encontrar fatores que pudessem prever a presença de doenças cardíacas.

Ponto de Referência (Benchmark): Centenas de artigos e projetos que usam este dataset como um ponto de partida para testar novos algoritmos. Ao usá-lo,  praticamos com o mesmo material que muitos profissionais e acadêmicos usaram para inovar na área.

### 2. As 📊 Variáveis Mais Relevantes para a Inteligência Artificial

Aqui estão as que eu considero mais relevantes para o projeto CardioIA, e o porquê:

❤️ Variáveis de Alto Impacto (Sinais Diretos do Coração)

💢 cp (Chest Pain Type - Tipo de Dor no Peito):

O que é: Descreve o tipo de dor que o paciente sente (ex: angina típica, angina atípica, dor não anginosa, assintomático).

Importância Clínica: Este é, talvez, o sintoma mais crítico. Para um médico, o tipo de dor no peito é um indicador fortíssimo da probabilidade de a causa ser cardíaca. A "angina típica", por exemplo, está diretamente ligada à falta de oxigênio no músculo do coração.

Para a IA: Esta é uma variável categórica poderosíssima. O modelo de IA aprenderá a associar certos tipos de dor a um risco muito maior de doença, espelhando o raciocínio diagnóstico de um cardiologista.

🏃‍♂️ 🔬 thalch (Maximum Heart Rate Achieved - Frequência Cardíaca Máxima Atingida):

O que é: O maior número de batimentos por minuto que o paciente atingiu durante um teste de esforço.

Importância Clínica: Um coração saudável consegue aumentar sua frequência para atender à demanda do exercício. Uma frequência máxima mais baixa do que o esperado para a idade pode indicar problemas na capacidade do coração de responder ao estresse.

Para a IA: É uma variável 🎯 numérica direta. Modelos de IA são ótimos em encontrar limiares e correlações. Eles podem descobrir que, a partir de certo ponto, uma 🏃‍♂️ 🔬 thalch mais baixa aumenta exponencialmente a probabilidade de doença, especialmente quando combinada com outras variáveis.

⚠️ exang (Exercise Induced Angina - Angina Induzida por Exercício):

O que é: Uma simples resposta 'sim' ou 'não' (True/False) se o paciente sentiu dor no peito (angina) durante o teste de esforço.

Importância Clínica: Se o esforço físico causa dor no peito, é um sinal clássico de que as artérias coronárias podem estar obstruídas, impedindo o fluxo de sangue necessário quando o coração trabalha mais.

Para a IA: Esta é uma variável binária (0 ou 1) com um poder de previsão altíssimo. É uma "bandeira vermelha" que o algoritmo aprenderá a levar muito a sério.

📉 oldpeak (ST Depression - Depressão do Segmento ST):

O que é: Este é um valor medido no eletrocardiograma (ECG) durante o teste de esforço. Ele mede o quanto o segmento "ST" do traçado do coração "caiu" em relação ao repouso.

Importância Clínica: A depressão do segmento ST é um dos indicadores mais fortes de isquemia miocárdica (sofrimento do músculo cardíaco por falta de sangue e oxigênio). É um achado técnico e objetivo do ECG.

Para a IA: Para um modelo de IA, esta variável 🎯 numérica é ouro puro. Ela quantifica um fenômeno elétrico diretamente ligado ao problema que queremos prever. Valores mais altos de 📉 oldpeak são fortes preditores de doença coronariana.

🔬 thal (Thallium Scan Results - Resultados do Teste de Talio):

O que é: O resultado de um exame de im👴 agem que mostra como o sangue flui para o coração. Os valores podem indicar um fluxo normal, um defeito fixo (uma área que nunca recebe sangue suficiente, como uma cicatriz de infarto) ou um defeito reversível (uma área que só sofre com falta de sangue durante o esforço).

Importância Clínica: É um diagnóstico visual do fluxo sanguíneo. Um "defeito reversível" é um sinal clássico de uma artéria significativamente obstruída.

Para a IA: Assim como o 💢 cp, é uma variável categórica com enorme poder de separação entre pacientes saudáveis e doentes.

🧩 Variáveis de Contexto (Fatores de Risco)

👴 age (Idade): O risco de doença cardíaca aumenta naturalmente com a idade.

⚧️ sex (Sexo): Homens e mulheres têm diferentes perfis de risco e idades de acometimento.

🩺 trestbps (Resting Blood Pressure - Pressão Arterial em Repouso): A hipertensão é um dos principais fatores de risco.

🥓 chol (Cholesterol - Colesterol): Níveis elevados de colesterol levam à formação de placas nas artérias.

A coluna 🎯 num é o alvo (🎯 target). É ela que indica o diagnóstico final (geralmente 0 para ausência de doença e valores maiores que 0 para diferentes graus de doença).

O objetivo do projeto será usar todas as outras variáveis para prever este resultado.




---

## 📝 Parte 2 – Dados Textuais (Processamento de Linguagem Natural - NLP)

Esta parte do projeto lida com dados não estruturados na forma de textos. O objetivo é extrair conhecimento e contexto que não estão presentes nos dados numéricos.

* **Descrição:** O corpus textual é composto por dois documentos: (1) um extrato técnico das Diretrizes Brasileiras de Hipertensão Arterial e (2) um texto informativo do Ministério da Saúde sobre o infarto, voltado para o público geral.
* **Objetivo Específico:** Criar uma base para treinar algoritmos de NLP capazes de **extrair entidades médicas (sintomas, tratamentos), classificar tópicos e entender a terminologia da cardiologia**.
* **Fontes Principais:**
    * **Texto Clínico:** [Diretrizes Brasileiras de Hipertensão Arterial - SciELO](https://www.scielo.br/j/abc/a/Z6m5gGNQCvrW3WLV7csqbqh/?lang=pt)
    * **Texto de Saúde Pública:** [Ministério da Saúde - Ataque Cardíaco](http://bvsms.saude.gov.br/dicas-em-saude/2779-ataque-cardiaco-infarto)
* **Ferramentas (Próximos Passos):** NLTK, spaCy, Scikit-learn.

---

## 🖼️ Parte 3 – Dados Visuais (Visão Computacional - VC)

Esta seção é dedicada à coleta de imagens de exames cardiológicos, a matéria-prima para treinar modelos de visão computacional.

* **Descrição:** Foram reunidas fontes de datasets públicos contendo imagens de três tipos principais de exames: Eletrocardiogramas (ECG), Raios-X de Tórax e Angiogramas Coronários.
* **Objetivo Específico:** Construir um banco de imagens para, futuramente, treinar redes neurais convolucionais (CNNs) para **detectar anomalias visuais**, como a presença de arritmias em um ECG ou o aumento da área cardíaca em um raio-X.
* **Fontes dos Datasets:**
    * **ECG:** [ECG Images Dataset on Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
    * **Raio-X de Tórax:** [NIH ChestX-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
    * **Angiograma:** [Coronary Angiography Database on Kaggle](https://www.kaggle.com/datasets/suyashlakhani/cardiovascular-disease-dataset)
* **Ferramentas (Próximos Passos):** OpenCV, TensorFlow, PyTorch.

---

