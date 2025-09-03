# 🩺 Projeto CardioIA: Base de Dados para Inteligência Artificial em Cardiologia

Sistema multimodal de coleta, organização e preparação de dados cardiovasculares para desenvolvimento de soluções de Inteligência Artificial em saúde.

## 📑 Sumário

- [🎯 Objetivos do Projeto](#-objetivos-do-projeto)
- [📊 Parte 1 – Dados Numéricos (IoT)](#-parte-1--dados-numéricos-iot)
  - [🏥 Origem dos Dados: Reais e com História Clínica](#-origem-dos-dados-reais-e-com-história-clínica)
  - [🔬 Variáveis Clinicamente Relevantes para IA](#-variáveis-clinicamente-relevantes-para-ia)
- [📝 Parte 2 – Dados Textuais (NLP)](#-parte-2--dados-textuais-nlp)
  - [📚 Corpus Textual Especializado](#-corpus-textual-especializado)
  - [🧠 Como Algoritmos de NLP Exploram os Textos](#-como-algoritmos-de-nlp-exploram-os-textos)
- [🖼️ Parte 3 – Dados Visuais (Visão Computacional)](#️-parte-3--dados-visuais-visão-computacional)
  - [🏥 Datasets de Imagens Médicas](#-datasets-de-imagens-médicas)
  - [👁️ Como Algoritmos de Visão Computacional Analisam Imagens](#️-como-algoritmos-de-visão-computacional-analisam-imagens)
- [🚀 Próximas Etapas do Projeto](#-próximas-etapas-do-projeto)


---

## 🎯 Objetivos do Projeto

- **Coletar Dados Multimodais:** Reunir dados numéricos, textuais e visuais de fontes públicas e confiáveis
- **Organizar e Limpar:** Estruturar os dados com foco em governança e qualidade
- **Análise Exploratória (EDA):** Extrair insights iniciais e compreender características dos datasets
- **Construir Base Sólida:** Criar alicerce preparado para treinamento de modelos de IA especializados

---

# 📊 Parte 1 – Dados Numéricos (IoT)

## 🏥 Origem dos Dados: Reais e com História Clínica

### Dataset Heart Disease UCI
- **📋 Fonte:** [Kaggle - Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **📁 Arquivo:** [heart_disease_uci.csv](https://drive.google.com/file/d/1Sam6ql00rbkOIR3cT5IARnHoySZr4d_7/view?usp=drive_link)

### ✅ Dados Reais de Pacientes
Este é um dos datasets mais respeitados na comunidade de Machine Learning médico. Os dados foram **coletados de pacientes reais** em quatro hospitais renomados:
- **Cleveland Clinic** (EUA)
- **Instituto Húngaro de Cardiologia**
- **Hospital Universitário de Zurique** (Suíça)
- **V.A. Medical Center** em Long Beach (EUA)

Cada linha representa a jornada clínica de um paciente real, coletada por médicos e pesquisadores com objetivo científico de identificar fatores preditivos de doenças cardíacas.

## 🔬 Variáveis Clinicamente Relevantes para IA

### ❤️ **Variáveis de Alto Impacto Clínico**

| Variável | Descrição | Relevância para IA |
|----------|-----------|-------------------|
| **🫀 cp** | Tipo de dor no peito | **Sintoma mais crítico** - IA aprende a associar tipos específicos de dor a risco cardiovascular, espelhando raciocínio diagnóstico médico |
| **🏃‍♂️ thalch** | Frequência cardíaca máxima atingida | **Capacidade funcional** - Modelos identificam limiares onde frequências baixas indicam comprometimento cardíaco |
| **⚠️ exang** | Angina induzida por exercício | **Sinal de obstrução** - Variável binária com alto poder preditivo para estenose coronariana |
| **📉 oldpeak** | Depressão do segmento ST no ECG | **Indicador objetivo** - Quantifica isquemia miocárdica, dados técnicos puros para algoritmos |
| **🔬 thal** | Resultado do teste de tálio | **Diagnóstico por imagem** - Mostra defeitos de perfusão, forte separador entre pacientes saudáveis e doentes |

### 🧩 **Variáveis de Contexto (Fatores de Risco)**

- **👴 age** - Risco aumenta naturalmente com idade
- **⚧️ sex** - Diferentes perfis de risco entre homens e mulheres
- **🩺 trestbps** - Pressão arterial (fator de risco estabelecido)
- **🥓 chol** - Colesterol (formação de placas arteriais)

### 🎯 **Por que são Relevantes para IA em Saúde?**

1. **Simulação de Raciocínio Clínico:** Algoritmos aprendem a combinar sintomas como médicos fazem
2. **Detecção de Padrões Complexos:** IA identifica combinações de variáveis que humanos podem não perceber
3. **Triagem Automatizada:** Sistemas podem priorizar casos de alto risco em tempo real
4. **Suporte à Decisão:** Complementam expertise médica com análise quantitativa objetiva

---

# 📝 Parte 2 – Dados Textuais (NLP)

## 📚 Corpus Textual Especializado

### Fontes dos Textos
- **📋 Texto Clínico:** [Diretrizes Brasileiras de Hipertensão - SciELO](https://www.scielo.br/j/abc/a/Z6m5gGNQCvrW3WLV7csqbqh/?lang=pt)
  - **Arquivo:** [texto_clinico_01.txt](docs/texto_clinico_01.txt)
- **🏥 Texto de Saúde Pública:** [Ministério da Saúde - Infarto](http://bvsms.saude.gov.br/dicas-em-saude/2779-ataque-cardiaco-infarto)
  - **Arquivo:** [texto_saudepublica_02.txt](docs/texto_saudepublica_02.txt)

## 🧠 Como Algoritmos de NLP Exploram os Textos

### 🔍 **1. Extração de Entidades Nomeadas (NER)**

**Como Funciona:** Treina algoritmos para identificar e classificar conceitos médicos específicos no texto.

**Aplicações nos Nossos Textos:**
- **Texto Clínico:** 
  - 🦠 Doenças: "Hipertensão Arterial", "risco cardiovascular"
  - 💊 Tratamentos: "diuréticos tiazídicos", "IECA", "dieta DASH"
  - 📊 Parâmetros: "pressão arterial (PA)"
- **Texto Saúde Pública:**
  - 🤒 Sintomas: "dor no peito", "falta de ar súbita", "suor frio"
  - ⚠️ Fatores de Risco: "diabetes", "colesterol elevado", "tabagismo"

### 📊 **2. Classificação de Tópicos (Topic Modeling)**

**Como Funciona:** Descobre automaticamente temas principais em grandes volumes de texto.

**Tópicos Identificáveis:**
- **💊 Tópico 1:** Farmacologia (medicamentos, IECA, diuréticos)
- **🥗 Tópico 2:** Prevenção (alimentação, exercício, estilo de vida)
- **🚑 Tópico 3:** Emergências (sintomas, atendimento, SAMU)

### 😊 **3. Análise de Sentimentos**

**Como Funciona:** Determina tom emocional do texto (positivo, negativo, neutro).

**Aplicação:** Nossos textos são informativos (neutros), mas técnica seria útil para analisar feedback de pacientes sobre tratamentos.

## 🎯 **Relevância para IA na Saúde**

### **📂 Estruturação de Prontuários**
- **Problema:** Prontuários médicos são majoritariamente texto livre
- **Solução IA:** NER extrai automaticamente sintomas, medicamentos e diagnósticos
- **Impacto:** Transforma dados não-estruturados em informação organizada para modelos preditivos

### **🤖 Triagem Inteligente**
- **Aplicação:** Chatbots médicos identificam sintomas-chave na descrição do paciente
- **Resultado:** Cálculo automático de nível de urgência e recomendações de atendimento

### **📊 Inteligência Operacional**
- **Uso:** Análise de milhares de relatórios médicos para identificar padrões
- **Benefício:** Descoberta de tendências, lacunas no conhecimento e oportunidades de melhoria

---

# 🖼️ Parte 3 – Dados Visuais (Visão Computacional)

## 🏥 Datasets de Imagens Médicas

| **Tipo de Exame** | **Dataset Fonte** | **Arquivo Local** |
|-------------------|-------------------|-------------------|
| **📈 ECG** | [Kaggle - Heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) | [Imagens ECG](https://drive.google.com/drive/folders/1FgA_sDv7jY53sg5c64__gpLbNS8qM9pN?usp=drive_link) |
| **🫁 Raio-X Tórax** | [NIH ChestX-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC) | [Imagens Raio-X](https://drive.google.com/drive/folders/1Hi_ObZ0Vgg4QNZA1rOccGyzkBFApNtsn?usp=drive_link) |
| **🩸 Angiograma** | [Kaggle - Cardiovascular](https://www.kaggle.com/datasets/suyashlakhani/cardiovascular-disease-dataset) | [Imagens Angiograma](https://drive.google.com/drive/folders/19wAqXGRr-Sx5h909gH4pZhW8MwC4SoOt?usp=drive_link) |

## 👁️ Como Algoritmos de Visão Computacional Analisam Imagens

### **🔍 1. Identificação de Bordas e Texturas**

**Como Funciona:** CNNs aplicam filtros para detectar características básicas (bordas, cantos, gradientes).

**Aplicações por Tipo de Exame:**
- **🫁 Raio-X:** Identifica bordas das costelas, contorno do coração, curva do diafragma
- **🩸 Angiograma:** Traça bordas dos vasos sanguíneos, distinguindo-os do fundo
- **📈 ECG:** Detecta picos agudos do complexo QRS e curvas das ondas P e T

**Importância:** Sem essa etapa, imagem é apenas ruído de pixels. Identificação de bordas permite **segmentação** - entender onde cada estrutura anatômica começa e termina.

### **🧩 2. Detecção de Padrões Complexos**

**Como Funciona:** Camadas profundas combinam características simples para formar padrões médicos reconhecíveis.

**Padrões Aprendidos:**
- **🫁 Raio-X:** Padrão completo da silhueta cardíaca e árvore brônquica
- **🩸 Angiograma:** Padrão de artéria saudável vs. artéria com obstrução
- **📈 ECG:** Sequência rítmica normal P-QRS-T que se repete

**Relevância:** Algoritmo evolui de ver "linhas e curvas" para reconhecer "coração", "artéria coronária" e "batimento cardíaco".

### **⚠️ 3. Reconhecimento de Anomalias**

**Como Funciona:** Após treinar com milhares de imagens normais e anormais, IA identifica quebras nos padrões esperados.

**Detecções Específicas:**
- **🫁 Raio-X:** 
  - **Anomalia:** Cardiomegalia (coração aumentado)
  - **Método:** Medição automática da proporção coração/tórax (>0.5 = anormal)
- **🩸 Angiograma:**
  - **Anomalia:** Estenose arterial (obstrução)
  - **Método:** Detecção de pontos onde bordas se aproximam e fluxo diminui
- **📈 ECG:**
  - **Anomalia:** Fibrilação atrial
  - **Método:** Identificação de intervalos R-R irregulares

## 🎯 **Importância para IA na Saúde**

### **🤝 Apoio à Decisão Clínica**
- **Função:** Sistema de triagem analisando centenas de exames
- **Benefício:** Destaca casos suspeitos para revisão médica prioritária
- **Impacto:** Otimiza tempo e reduz erros por fadiga

### **📏 Quantificação Objetiva**
- **Evolução:** De "parece um pouco aumentado" para medições precisas
- **Vantagem:** Acompanhamento reprodutível da progressão da doença
- **Aplicação:** Monitoramento quantitativo ao longo do tempo

### **🔬 Detecção Precoce**
- **Capacidade:** IA detecta mudanças sutis imperceptíveis ao olho humano
- **Resultado:** Diagnósticos mais precoces em estágios iniciais
- **Impacto:** Melhor prognóstico e opções de tratamento

---

## 🚀 Próximas Etapas do Projeto

1. **📈 Análise Exploratória** detalhada de cada modalidade
2. **🔧 Pré-processamento** e limpeza dos dados
3. **🤖 Desenvolvimento** de modelos especializados:
   - Machine Learning para dados numéricos
   - NLP para processamento textual
   - CNNs para análise visual
4. **🔗 Integração** multimodal para sistema unificado

---
