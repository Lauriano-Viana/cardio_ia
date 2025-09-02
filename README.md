# 🩺 Projeto CardioIA: Base de Dados para Inteligência Artificial em Cardiologia

Bem-vindo ao repositório do Projeto CardioIA. Este projeto representa a fase fundamental de **coleta, organização e preparação de uma base de dados multimodal** (numérica, textual e visual) para o desenvolvimento de futuras soluções de Inteligência Artificial aplicadas à saúde cardiovascular.

Este documento serve como guia central, explicando os objetivos, as fontes de dados e a estrutura de cada componente do projeto.

## 🎯 Objetivos do Projeto

* **Coletar Dados Multimodais:** Reunir dados numéricos, textuais e visuais de fontes públicas e confiáveis para criar um repositório rico e diversificado.
* **Organizar e Limpar:** Estruturar os dados de forma lógica e aplicar técnicas de limpeza para garantir a qualidade e a consistência, com foco em governança de dados.
* **Análise Exploratória (EDA):** Realizar uma análise inicial para extrair os primeiros insights e compreender as características de cada conjunto de dados.
* **Construir uma Base Sólida:** Criar um alicerce de dados preparado para as próximas fases de um projeto de IA, como treinamento de modelos preditivos, sistemas de NLP e algoritmos de visão computacional.

---

# 📊 Parte 1 – Dados Numéricos (Análise Preditiva)

Esta seção foca em dados estruturados, tipicamente provenientes de exames clínicos e medições de pacientes, que poderiam ser coletados por dispositivos (IoT) ou inseridos em prontuários eletrônicos.

* **Descrição:** O conjunto de dados contém 14 atributos clínicos de pacientes, como idade, sexo, nível de colesterol, pressão arterial e resultados de testes de esforço.
* **Objetivo Específico:** Preparar os dados para treinar um modelo de Machine Learning capaz de **prever a presença ou ausência de doença cardíaca** com base nesses atributos.
* **Fonte Principal:**
    * **Dataset:** Heart Disease UCI
    * **Link:** [Kaggle - Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
    * **🔗 Link para o Arquivo:** 
        **[heart_disease_uci.csv](https://drive.google.com/file/d/1Sam6ql00rbkOIR3cT5IARnHoySZr4d_7/view?usp=drive_link)**


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

# 📝 Parte 2 – Dados Textuais (Processamento de Linguagem Natural - NLP)

Esta parte do projeto lida com dados não estruturados na forma de textos. O objetivo é extrair conhecimento e contexto que não estão presentes nos dados numéricos.

* **Descrição:** O corpus textual é composto por dois documentos: (1) um extrato técnico das Diretrizes Brasileiras de Hipertensão Arterial e (2) um texto informativo do Ministério da Saúde sobre o infarto, voltado para o público geral.
* **Objetivo Específico:** Criar uma base para treinar algoritmos de NLP capazes de **extrair entidades médicas (sintomas, tratamentos), classificar tópicos e entender a terminologia da cardiologia**.
* **Fontes Principais:**
    * **Texto Clínico:** [Diretrizes Brasileiras de Hipertensão Arterial - SciELO](https://www.scielo.br/j/abc/a/Z6m5gGNQCvrW3WLV7csqbqh/?lang=pt)
    * **Texto de Saúde Pública:** [Ministério da Saúde - Ataque Cardíaco](http://bvsms.saude.gov.br/dicas-em-saude/2779-ataque-cardiaco-infarto)
    * **🔗 Link para o Arquivo:**
     
        **[texto_clinico_01.txt](docs/texto_clinico_01.txt)**
        
        **[texto_saudepublica_02.txt](docs/texto_saudepublica_02.txt)**


## 🧠 Como Algoritmos de NLP Exploram os Textos e Por Que Isso é Relevante

🧩 **Processamento de Linguagem Natural (PLN)**, ou *NLP* em inglês, é um campo da IA que ensina os computadores a entender e processar a linguagem humana. Pense nele como um tradutor entre o caos da nossa comunicação e a ordem lógica dos números que um computador entende.

Vamos ver três técnicas poderosas e como elas se aplicam aos nossos textos:

---

### 🔍 1. Extração de Entidades Nomeadas (Named Entity Recognition - NER)

- **O que é?** É a técnica de treinar um algoritmo para identificar e classificar "entidades" — palavras ou grupos de palavras que representam conceitos específicos e predefinidos. É como dar ao computador "marcadores de texto" de cores diferentes para cada tipo de informação. A "extração de sintomas" que você mencionou é um caso de uso perfeito para NER.

#### 📌 Aplicação nos nossos textos:

- No `texto_clinico_01.txt`, um modelo de NER treinado para a área da saúde poderia extrair:
  - **🦠 Doenças/Condições:** Hipertensão Arterial, risco cardiovascular
  - **💊 Tratamentos (Não Medicamentosos):** alimentação saudável, dieta DASH, exercícios aeróbicos
  - **💉 Medicamentos/Classes de Fármacos:** diuréticos tiazídicos, IECA, BRA
  - **📊 Parâmetros Clínicos:** pressão arterial (PA)

- No `texto_saudepublica_02.txt`, ele extrairia:
  - **🦠 Doenças/Condições:** Infarto agudo do miocárdio, aterosclerose
  - **🤒 Sintomas:** dor ou desconforto intenso na região do peito, falta de ar súbita, suor frio, náuseas, tontura
  - **⚠️ Fatores de Risco:** hipertensão arterial, diabetes, colesterol elevado, tabagismo

#### ✅ Por que é Relevante para a Saúde?

- **📂 Estruturação de Dados:** Prontuários médicos são, em grande parte, texto livre. Um sistema de IA com NER pode "ler" a anotação de um médico e automaticamente extrair os sintomas, medicamentos prescritos e diagnósticos, transformando o texto não estruturado em dados organizados. Isso economiza um tempo imenso e alimenta os modelos preditivos com informações muito mais ricas.

- **🤖 Triagem Inteligente:** Um chatbot de triagem poderia usar NER para identificar sintomas-chave na descrição de um paciente e, com base nisso, calcular um nível de urgência e recomendar a busca por atendimento.

---

### 🧩 2. Classificação de Tópicos (Topic Modeling)

- **O que é?** É uma técnica não supervisionada que analisa um grande volume de textos e descobre, automaticamente, os principais temas ou "tópicos" abordados. O algoritmo agrupa palavras que frequentemente aparecem juntas para formar esses tópicos.

#### 📌 Aplicação nos nossos textos:

- Se tivéssemos centenas de documentos como os nossos, um modelo de classificação de tópicos poderia identificar, por exemplo:

  - **📌 Tópico 1:** (palavras-chave: tratamento, medicamentoso, fármacos, IECA, diuréticos, PA) → **Tópico: Farmacologia da Hipertensão**
  - **🥗 Tópico 2:** (palavras-chave: alimentação, exercício, sal, peso, dieta, estilo de vida) → **Tópico: Mudanças no Estilo de Vida e Prevenção**
  - 🚑 **Tópico 3:** (palavras-chave: sintomas, dor, peito, infarto, atendimento, emergência, SAMU) → **Tópico: Reconhecimento e Resposta a Emergências Cardíacas**

#### ✅ Por que é Relevante para a Saúde?

- **📚 Análise de Literatura Científica:** Pesquisadores podem usar essa técnica para analisar milhares de artigos científicos e identificar rapidamente as principais linhas de pesquisa, tendências emergentes ou lacunas no conhecimento sobre uma doença.

- **🏥 Inteligência Operacional em Hospitais:** Um hospital poderia analisar milhares de relatos de incidentes ou reclamações de pacientes para descobrir tópicos recorrentes (ex: "demora no atendimento", "problemas com medicação"), permitindo que a gestão atue de forma mais direcionada.

---

### 😊 3. Análise de Sentimentos

- **O que é?** É o processo de determinar o tom emocional ou a polaridade (positivo, negativo, neutro) de um texto.

#### 📌 Aplicação nos nossos textos:

- Os textos que selecionamos são informativos e, portanto, seriam classificados como **neutros**.

- No entanto, imagine que tivéssemos um terceiro tipo de texto: *reviews de pacientes sobre o atendimento em uma clínica cardiológica*.

  - *"O Dr. Silva foi incrível, explicou tudo com calma e me senti muito seguro."* → **Sentimento: Positivo** 😊
  - *"Esperei três horas para ser atendido e a recepção foi muito desorganizada."* → **Sentimento: Negativo** 😠

#### ✅ Por que é Relevante para a Saúde?

- **👥 Experiência do Paciente (Patient Experience):** Hospitais e clínicas podem monitorar mídias sociais ou formulários de feedback para medir a satisfação do paciente em tempo real, identificando pontos de atrito e melhorando a qualidade do serviço.

- **🌍 Saúde Pública:** Governos podem usar a análise de sentimentos para medir a percepção pública sobre campanhas de vacinação, políticas de saúde ou o medo associado a uma epidemia, ajustando a comunicação para ser mais eficaz.

---

### ✅ Conclusão

Em resumo, ao integrar o NLP, seu projeto **CardioIA** evolui. Ele deixa de ser apenas um sistema que olha para números e passa a ser um sistema que pode **ler, interpretar e contextualizar** a vasta quantidade de informação textual que permeia toda a área da saúde. É um passo crucial para construir soluções de IA verdadeiramente inteligentes e úteis.

---


# 🖼️ Parte 3 – Dados Visuais (Visão Computacional - VC)

Esta seção é dedicada à coleta de imagens de exames cardiológicos, a matéria-prima para treinar modelos de visão computacional.

* **Descrição:** Foram reunidas fontes de datasets públicos contendo imagens de três tipos principais de exames: Eletrocardiogramas (ECG), Raios-X de Tórax e Angiogramas Coronários.
* **Objetivo Específico:** Construir um banco de imagens para, futuramente, treinar redes neurais convolucionais (CNNs) para **detectar anomalias visuais**, como a presença de arritmias em um ECG ou o aumento da área cardíaca em um raio-X.
* **Fontes dos Datasets:**
    * **ECG:** [ECG Images Dataset on Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
    * **Raio-X de Tórax:** [NIH ChestX-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
    * **Angiograma:** [Coronary Angiography Database on Kaggle](https://www.kaggle.com/datasets/suyashlakhani/cardiovascular-disease-dataset)

Claro! Aqui está o conteúdo do arquivo convertido para **Markdown** com ícones para incluir em um `README.md`:

---

## 👁️ Como Algoritmos de Visão Computacional Analisam Imagens Médicas

🧠 Fundamentalmente, um computador não vê uma imagem como nós. Ele a vê como uma gigantesca matriz de números, onde cada número representa a intensidade de um pixel. O trabalho da Visão Computacional (VC) é transformar essa matriz de números em insights clínicos. O principal "motor" por trás disso hoje são as **Redes Neurais Convolucionais (CNNs)**.

👨‍⚕️ Imagine uma CNN como um médico residente em treinamento. O processo de aprendizado ocorre em etapas, muito similar às técnicas que você mencionou:

---

### 🔍 1. Identificação de Bordas e Texturas (As Lições Básicas)

- **Como Funciona?** As primeiras camadas de uma CNN funcionam como um scanner de características básicas. Elas aplicam uma série de "filtros" (chamados de *kernels*) que varrem a imagem para encontrar elementos fundamentais:
  - **📏 Bordas:** Linhas verticais, horizontais e diagonais.
  - **🔲 Cantos:** Pontos onde as bordas se encontram.
  - **🎨 Gradientes:** Mudanças de cor ou brilho.
  - **🧵 Texturas:** Padrões repetitivos.

#### 📌 Aplicação nas Nossas Imagens:

- **🫁 Raio-X Torácico:** O algoritmo primeiro aprende a identificar as bordas nítidas das costelas, a curva suave do diafragma e o contorno que define a **silhueta do coração**.
- **🩸 Angiograma:** Ele aprende a traçar as bordas dos vasos sanguíneos, distinguindo-os do fundo.
- **📈 Eletrocardiograma (ECG):** Ele identifica as bordas e picos agudos do complexo QRS e as curvas mais suaves das ondas P e T.

#### ✅ Importância:
Sem essa etapa, a imagem é apenas um ruído de pixels. A identificação de bordas e padrões básicos é o que permite ao algoritmo **segmentar** a imagem, ou seja, entender onde um objeto (como o coração) termina e outro (como o pulmão) começa. É o alicerce de toda a compreensão visual.

---

### 🧩 2. Detecção de Padrões (Combinando as Lições)

- **Como Funciona?** As camadas mais profundas da CNN pegam as características simples da etapa anterior e aprendem a combiná-las para formar padrões mais complexos. É como aprender que um círculo com dois pontos e uma linha curva abaixo forma um "rosto".

#### 📌 Aplicação nas Nossas Imagens:

- **🫁 Raio-X Torácico:** O algoritmo combina as bordas detectadas para reconhecer o padrão completo da "silhueta cardíaca" ou a "árvore brônquica" dos pulmões.
- **🩸 Angiograma:** Ele aprende o padrão de uma artéria saudável (larga e com fluxo contínuo) em oposição a uma artéria doente.
- **📈 Eletrocardiograma (ECG):** Ele aprende a reconhecer o padrão rítmico completo de um "batimento sinusal normal" (a sequência P-QRS-T que se repete).

#### ✅ Importância:
É aqui que o reconhecimento de objetos acontece. O algoritmo agora não vê apenas linhas e curvas, mas sim "coração", "artéria coronária" e "batimento cardíaco". Essa capacidade de reconhecer estruturas anatômicas é a base para qualquer diagnóstico automatizado.

---

### ⚠️ 3. Reconhecimento de Anomalias (O Diagnóstico)

- **Como Funciona?** Depois de ter aprendido com milhares de imagens normais e anormais, a rede neural se torna uma especialista em "o que é normal". Uma anomalia é, essencialmente, uma quebra no padrão esperado. A etapa final é classificar a imagem ou uma parte dela com base nesses padrões.

#### 📌 Aplicação nas Nossas Imagens:

- **🫁 Raio-X Torácico:** O algoritmo mede o padrão da "silhueta cardíaca" que ele detectou. Se a proporção entre a largura do coração e a largura do tórax exceder um limiar aprendido (geralmente > 0.5), ele classifica a imagem como tendo **Cardiomegalia (coração aumentado)**, uma anomalia crítica.
- **🩸 Angiograma:** Ao seguir o padrão de uma artéria, o algoritmo detecta um ponto onde as bordas se aproximam drasticamente e o fluxo de contraste diminui. Ele reconhece essa quebra de padrão como uma **estenose (obstrução)**.
- **📈 Eletrocardiograma (ECG):** O algoritmo compara o padrão do batimento do paciente com o padrão "normal" que ele aprendeu. Se os picos R estiverem irregularmente espaçados, ele pode classificar a anomalia como **"Fibrilação Atrial"**.

* **🔗 Links para as Imagens:**
    * **[Imagens de Raio-X Torácico](https://drive.google.com/drive/folders/1Hi_ObZ0Vgg4QNZA1rOccGyzkBFApNtsn?usp=drive_link)**
    * **[Imagens de Angiograma Coronário](https://drive.google.com/drive/folders/19wAqXGRr-Sx5h909gH4pZhW8MwC4SoOt?usp=drive_link)**
    * **[Imagens de Eletrocardiograma (ECG)](https://drive.google.com/drive/folders/1FgA_sDv7jY53sg5c64__gpLbNS8qM9pN?usp=drive_link)**

#### ✅ Importância:
Este é o resultado final que impacta diretamente o cuidado ao paciente.

- **🤝 Apoio à Decisão Clínica:** A IA pode funcionar como um sistema de triagem, analisando centenas de exames e destacando os mais suspeitos para que o médico os revise primeiro. Isso otimiza o tempo e reduz o risco de erros por fadiga.
- **📊 Quantificação Objetiva:** Em vez de um médico dizer "parece um pouco aumentado", a IA pode fornecer uma medição precisa e reprodutível, permitindo um acompanhamento mais acurado da progressão da doença ao longo do tempo.
- **🔬 Detecção Precoce:** Muitas vezes, as anomalias começam com mudanças de textura muito sutis em uma imagem, que o olho humano pode não notar. Uma IA treinada pode captar esses sinais precoces, levando a um diagnóstico mais rápido.

---

### ✅ Conclusão

Em resumo, a Visão Computacional no seu projeto **CardioIA** transforma imagens de meros dados visuais em fontes ricas de informação diagnóstica, permitindo análises em uma escala e com uma precisão que complementam e potencializam a capacidade do especialista humano.

---


