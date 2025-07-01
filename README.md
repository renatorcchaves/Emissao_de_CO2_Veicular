# Projeto de Emissão de CO<sub>2</sub> de Veículos

O projeto atual tem como objetivo, a partir de uma base de dados do governo canadense, entender como varia a emissão de CO<sub>2</sub> de veículos de acordo com features como marca e modelo do carro, tipo de combustível, ano de fabricação, tipo do carro, transmissão, número de cilindros do motor e a autonomia do carro, isto é, quantos litros de combustível o carro consome para cada 100km percorridos em meio urbano ou rodoviário.

Base retirada do site do [governo canadense](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64).

## **Análise Exploratória** 
- Os dados retirados do link acima foram unificados, tratados e analisados através do arquivo [EDA_Analise_Exploratoria](notebooks/01_EDA_Analise_Exploratoria.ipynb). Com diversos gráficos já pudemos ter uma ideia inicial de como cada feature influenciava a nossa variável alvo (emissão de CO2).

## **Modelo de Regressão** 
- Com os dados tratados, através do arquivo [Comparando_Modelos_Regressão](notebooks/02_Comparando_Modelos_Regressao.ipynb) foram testados diversos modelos e regressão - como LinearRegression, Ridge, Lasso, DecisionTreeRegressor, LightGBMRegressor, XGBRegressor, KNeighborsRegressor e LinearSVR - para entender qual deles tinha melhores resultados na previsão da emissão de CO<sub>2</sub> para cada veículo. 
- Foi tomado o cuidado de fazer o preprocessamento das colunas categóricas ordinais e não ordinais, e da normalização dos dados de features numéricas e da variável alvo através de métodos como PowerTransform e QuantileTransform. 
- Tendo definido o melhor dos modelos testados acima, o que apresentou melhores resultados (XGBRegressor) passou pela otimização de parametros através de ferramentas como GridSearchCV. 
- Busquei entender quais features mais influenciavam a variável alvo através de métodos como Permutation Importance e Feature Importance (*imagem abaixo*).
- Analisei o desempenho (erro) que esse melhor modelo gerou quando comparado entre uma base de treino e teste através de métodos como PredictionErrorDisplay (*imagem abaixo*).
- O melhor modelo otimizado foi exportado com o nome [XGB_Regressor](modelos/xgb_regressor.joblib) para ser usado na página do Streamlit.

**Features Mais Importantes**

<img src="relatorios/imagens/Permutation Importance.png" title="Título" height="400"/>


**Desempenho do Modelo**
![Desempenho do Modelo XGBRegressor](relatorios/imagens/Resultados%20Treino%20x%20Teste%20-%20Modelo%20XGBRegressor.png)


## **Streamlit** - [![Streamlit App](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)](https://emissaodeco2veicular-projeto-renato.streamlit.app/)

*Para entrar no aplicativo do Streamlit deste projeto basta clicar no ícone acima*

- Para melhorar a interatividade com os dados deste projeto, foi criado um código para ser usado em uma página do Streamlit por meio do arquivo "home_streamlit.py".
- A página do possui 2 abas, uma para análise exploratória dos dados e outra para realizar a previsão da emissão de CO2 com base em algumas características do veículo
- *Análise Exploratória dos Dados*: onde existem tabelas com filtros, gráficos como treemap, gráficos de barra, scatterplot feitos através da biblioteca Plotly para permitir melhor interação do usuário para entender como a emissão de CO2 varia com cada feature dos dados.
- *Modelo de Previsão de CO<sub>2</sub>*: fazendo uso do modelo exportado citado acima, o usuário pode inputar algumas informações genéricas de qualquer carro (ano, tipo de carro, tamanho do motor, numero de cilindros, transmissão, tipo de combustível, consumo urbano, rodoviário e combinado) e há um botão para fornecer uma estimativa da emissão de CO2 do veículo com base nesses dados inputados. 


## Organização do projeto

```
├── .env               <- Arquivo de variáveis de ambiente (não versionar)
├── .gitignore         <- Arquivos e diretórios a serem ignorados pelo Git
├── ambiente.yml       <- O arquivo de requisitos para reproduzir o ambiente de análise
├── LICENSE            <- Licença de código aberto se uma for escolhida
├── README.md          <- README principal para desenvolvedores que usam este projeto.
|
├── dados              <- Arquivos de dados para o projeto.
|
├── modelos            <- Modelos treinados, otimizados e extraídos do projeto.
|
├── notebooks          <- Cadernos Jupyter onde foi desenvolvido o projeto
│
|   └──src             <- Código-fonte para uso neste projeto.
|      │
|      ├── __init__.py   <- Torna um módulo Python
|      ├── config.py     <- Configurações básicas do projeto
|      └── models.py     <- Scripts com fórmulas para criação, treino, e verificação dos resultados dos modelos
|      └── graficos.py   <- Scripts para criar visualizações exploratórias e orientadas a resultados
|      └── auxiliares.py <- Scripts para criar dataframe dos coeficientes do modelo escolhido
|
|
├── referencias        <- Dicionários de dados, manuais e todos os outros materiais explicativos.
|
├── relatorios         <- Análises geradas em HTML, PDF, LaTeX, etc.
│   └── imagens        <- Gráficos e figuras gerados para serem usados em relatórios



