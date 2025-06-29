import streamlit as st
import pandas as pd
import plotly.express as px
from joblib import load

from notebooks.src.config import DADOS_CONSOLIDADOS, DADOS_TRATADOS, MODELO_FINAL, MODELO_FINAL_XGB

#----------------------------------------------------------------------------------------------------------------------------------------
# Carregando dados e modelos

@st.cache_data
def carregar_dados(arquivo):
    return pd.read_parquet(arquivo)

@st.cache_resource
def carregar_modelo(arquivo):
    return load(arquivo)

df_consolidado = carregar_dados(DADOS_CONSOLIDADOS)
df_tratados = carregar_dados(DADOS_TRATADOS)
modelo_xgb = carregar_modelo(MODELO_FINAL_XGB)

colunas_para_retirar = [
    'co2_rating',
    'smog_rating',
    'combined_mpg',
    'engine_size_l',
    'cylinders',
    'city_l_100_km',
    'highway_l_100_km'
]

df_consolidado = df_consolidado.drop(columns=colunas_para_retirar, errors='ignore')

df_consolidado = df_consolidado[[       # Ordenando a sequência das colunas
        "model_year", "make", "model", "co2_emissions_g_km", "fuel_type", "vehicle_class", "combined_l_100_km"
]]

#----------------------------------------------------------------------------------------------------------------------------------------
# Criando abas dentro do Streamlit e os componentes do site

st.title("_Projeto de Emissão de CO2 Veicular_")
st.write("O projeto foi realizado com uma base de dados de veiculos canadentes, possuindo 2 páginas no Streamlit, uma delas para consultar os dados da base original consolidada dos ultimos anos, e outra em que são informados alguns dados do veículo e há um modelo de regressão que prevê a emissão de CO2 com base nas características informadas.")
st.write("Fonte: Base retirada do site do [governo canadense](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64).")

aba1, aba2 = st.tabs(["#### Análise Exploratória dos Dados", "#### Modelo de Previsão de CO2"])

with aba1:

    st.write("#### Análise dos Dados de Veículos Canadenses")
    
    # Para criar filtros num dataframe seguir documentação conforme link do Blog do Streamlit
    # https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/

    from pandas.api.types import (
        is_categorical_dtype,
        is_datetime64_any_dtype,
        is_numeric_dtype,
        is_object_dtype,
    )

    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        modify = st.checkbox("Adicionar filtros")

        if not modify:
            return df

        df = df.copy()

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]

        return df

    # Usando o df_consolidado para criar filtro e exibir resultados da base original usada na Análise Exploratória
    df_filter = filter_dataframe(df_consolidado)
    st.dataframe(
        df_filter.style.background_gradient(
            subset=["co2_emissions_g_km", "combined_l_100_km"],           # Definindo colunas a serem coloridas
            cmap='RdYlGn_r',                                              # Escolhendo o mapa de cores (poderia usar o 'coolwarm' tbm)
        ))
    

# Criando gráficos do Plotly Express
    cmin, cmax = (df_consolidado['co2_emissions_g_km'].min(), df_consolidado['co2_emissions_g_km'].max())


    #-------> Gráfico 6
    fig6 = px.treemap(
        df_consolidado,
        path=[
            px.Constant('co2_emissions_g_km'), 'make', 'vehicle_class', 'model'  # 'fuel_type', 'model_year'
        ],
        color='co2_emissions_g_km',
        color_continuous_scale='rdylgn_r',
        range_color=[cmin, cmax],
        title = 'Treemap de Emissão de CO<sub>2</sub>',
        labels={'co2_emissions_g_km': 'Emissão de CO<sub>2</sub> (g/100 km)'},
        hover_data = {'co2_emissions_g_km': ":.1f"}
    )
    st.plotly_chart(fig6)

    #-------> Gráfico 1
    fig1 = px.bar(
        df_consolidado[['make', 'co2_emissions_g_km']].groupby('make').mean().reset_index(),   
        x='make',
        y='co2_emissions_g_km',
        title='Média de Emissão de CO<sub>2</sub> por Fabricante (g/km)',
        color='co2_emissions_g_km',
        color_continuous_scale='rdylgn_r',
        hover_data={'co2_emissions_g_km': ':.1f'},            # Formatação do hover_data para exibir com uma casa decimal
        
    )
                # OBS: Nos plotly_express precisa passar o dataframe já agrupados (com groupby e função de agregação), não só o dataframe
                # OBS2: Se não usar o reset_index() o gráfico não funciona, pois o Plotly Express não aceita o índice como eixo x
                # OBS3: hover é o movimento de passar o mouse sobre o gráfico, e o hover_data é usado para definir quais dados serão exibidos

    fig1.update_xaxes(categoryorder='total descending')      # Ordenando o eixo x pela média de emissão de CO2 de maneira decrescente
    fig1.data[0].update(marker_cmin=cmin, marker_cmax=cmax)  # Definindo os valores mínimo e máximo do eixo y (para a cor)
    fig1.add_hline(                                          # Adicionando linha horizontal no gráfico com valor da média
        y=df_consolidado['co2_emissions_g_km'].mean(),
        line_dash='dot', 
        line_color='purple'
    )
    fig1.add_annotation(                                     # Colocando comentário do valor da média acima da linha horizontal 
        x=0.95,              
        y=df_consolidado['co2_emissions_g_km'].mean(),
        text = f"Média: {df_consolidado['co2_emissions_g_km'].mean():.2f} g/km",
        xref='paper',
        yshift=5
    )
    st.plotly_chart(fig1)              # Criando gráfico no Streamlit com a figura do Plotly Express


    #-------> Gráfico 2
    fig2 = px.bar(
        df_consolidado[['vehicle_class', 'co2_emissions_g_km']].groupby('vehicle_class').mean().reset_index(),   
        x='vehicle_class',
        y='co2_emissions_g_km',
        title='Média de Emissão de CO<sub>2</sub> por Tipo de Veículo (g/km)',
        color='co2_emissions_g_km',
        color_continuous_scale='rdylgn_r',
        hover_data={'co2_emissions_g_km': ':.1f'},           # Formatação do hover_data para exibir com uma casa decimal
        range_color = [cmin, cmax]                           # Pra manter a mesma escala de cores padronizada de antes
    )
    fig2.update_xaxes(categoryorder='total descending')      # Ordenando o eixo x pela média de emissão de CO2 de maneira decrescente
    fig2.data[0].update(marker_cmin=cmin, marker_cmax=cmax)  # Definindo os valores mínimo e máximo do eixo y (para a cor)
    fig2.add_hline(                                          # Adicionando linha horizontal no gráfico com valor da média
        y=df_consolidado['co2_emissions_g_km'].mean(),
        line_dash='dot', 
        line_color='purple'
    )
    fig2.add_annotation(                                     # Colocando comentário do valor da média acima da linha horizontal 
        x=0.95,              
        y=df_consolidado['co2_emissions_g_km'].mean(),
        text = f"Média: {df_consolidado['co2_emissions_g_km'].mean():.2f} g/km",
        xref='paper',
        yshift=5
    )
    st.plotly_chart(fig2)              # Criando gráfico no Streamlit com a figura do Plotly Express    


    #-------> Gráfico 3
    fig3 = px.bar(
        df_consolidado[['model_year', 'co2_emissions_g_km']].groupby('model_year').mean().reset_index(),   
        x='model_year',
        y='co2_emissions_g_km',
        title='Média de Emissão de CO<sub>2</sub> por Ano (g/km)',
        color='co2_emissions_g_km',
        color_continuous_scale='rdylgn_r',
        hover_data={'co2_emissions_g_km': ':.1f'},            # Formatação do hover_data para exibir com uma casa decimal
        range_color = [cmin, cmax]                            # Pra manter a mesma escala de cores padronizada de antes
    )
    #fig3.update_xaxes(categoryorder='total descending')     # COMENTANDO PORQUE NÃO FAZ SENTIDO OS ANOS SEREM ORDENADORES DE MANEIRA DECRESCENTE
    fig3.data[0].update(marker_cmin=cmin, marker_cmax=cmax)  # Definindo os valores mínimo e máximo do eixo y (para a cor)
    fig3.add_hline(                                          # Adicionando linha horizontal no gráfico com valor da média
        y=df_consolidado['co2_emissions_g_km'].mean(),
        line_dash='dot', 
        line_color='purple'
    )
    fig3.add_annotation(                                     # Colocando comentário do valor da média acima da linha horizontal 
        x=0.95,              
        y=df_consolidado['co2_emissions_g_km'].mean(),
        text = f"Média: {df_consolidado['co2_emissions_g_km'].mean():.2f} g/km",
        xref='paper',
        yshift=5
    )
    st.plotly_chart(fig3)              # Criando gráfico no Streamlit com a figura do Plotly Express (se não a figura não aparecerá)       


    #-------> Gráfico 4
    fig4 = px.scatter(
        df_consolidado,
        x='combined_l_100_km',
        y='co2_emissions_g_km',
        color='fuel_type',                                                 # Mudando as cores de acordo com o tipo de combustível
        color_discrete_sequence=px.colors.qualitative.Set1,                # Definindo o mapa de cores a ser usado
        opacity=0.5,
        title='Emissão de CO<sub>2</sub> X Consumo Combinado - Tipo de Combustível',
        labels={                                                           # Fazendo um "de:para" da legenda pro português
            'combined_l_100_km': "Consumo Combinado (l/100 km)",
            'co2_emissions_g_km': 'Emissão de CO<sub>2</sub> (g/100 km)'
        }
    )
    fig4.update_layout(                                                     # Alterando posicionamento da legenda
        legend=dict(
            title="Tipo de Combustível",
            orientation='h',                                                # Na hora horizontal
            yanchor='bottom',                                               # Alinhado com a parte inferior
            y=0.95,                                                         # a 95% da altura da imagem começando da parte inferior
            xanchor='right',                                                # Alinhado a direita
            x=1,                                                            # Na hora horizontal
        )
    )
    st.plotly_chart(fig4)              # Criando gráfico no Streamlit com a figura do Plotly Express (se não a figura não aparecerá)    


    #-------> Gráfico 5
    fig5 = px.scatter(
        df_consolidado,
        x='combined_l_100_km',
        y='co2_emissions_g_km',
        color='vehicle_class',                                             # Mudando as cores de acordo com o tipo de combustível
        color_discrete_sequence=px.colors.qualitative.Light24,             # Definindo o mapa de cores a ser usado
        opacity=0.5,
        title='Emissão de CO<sub>2</sub> X Consumo Combinado - Classe de Veículo',
        labels={                                                           # Fazendo um "de:para" da legenda pro português
            'combined_l_100_km': "Consumo Combinado (l/100 km)",
            'co2_emissions_g_km': 'Emissão de CO<sub>2</sub> (g/100 km)'
        }
    )
    st.plotly_chart(fig5)              # Criando gráfico no Streamlit com a figura do Plotly Express (se não a figura não aparecerá)  

with aba2:

    st.write("#### Prever Emissão de CO2 do Veículo")

    # Definindo variáveis conforme dataframe tratado
    ano = sorted(df_tratados['model_year'].unique())
    transmissao = sorted(df_tratados['transmission'].unique())  
    combustivel = sorted(df_tratados['fuel_type'].unique())
    tipo_veiculo = sorted(df_tratados['vehicle_class_grouped'].unique())
    tamanho_motor = sorted(df_tratados['engine_size_l_class'].unique())
    cilindros = sorted(df_tratados['cylinders_class'].unique())

    colunas_slider = ['city_l_100_km', 'highway_l_100_km', 'combined_l_100_km',]

    colunas_slider_min_max = {
        coluna: {
            "min_value": df_tratados[coluna].min(), 
            "max_value": df_tratados[coluna].max()}
        for coluna in colunas_slider} 

    # Deixando todos os widgets dentro de um formulário (pra não ficar recarregando a página toda vez que um widget é alterado)
    with st.form(key='formulario'):          

        coluna_esquerda, coluna_direita = st.columns(2)
        
        with coluna_esquerda:
            widget_ano = st.selectbox("Ano", ano)
            widget_tamanho_motor = st.selectbox("Tamanho do Motor", tamanho_motor)
            widget_transmissao = st.radio("Transmissão", transmissao)
        with coluna_direita:
            widget_classe_veiculo = st.selectbox("Classe do Veículo", tipo_veiculo)
            widget_cilindros = st.selectbox("Quantidade de Cilindros", cilindros)   
            widget_combustivel = st.radio("Tipo de Combustível", combustivel)

        widget_consumo_urbano = st.slider("Consumo Urbano (l/100km)", **colunas_slider_min_max['city_l_100_km'])
        widget_consumo_rodovia = st.slider("Consumo Rodovia (l/100km)", **colunas_slider_min_max['highway_l_100_km'])
        widget_consumo_combinado = st.slider("Consumo Combinado (l/100km)", **colunas_slider_min_max['combined_l_100_km'])

        botao_previsao = st.form_submit_button("Prever Emissao de CO2")
        
    # Reunindo dados inputados no Streamlit para fazer Previsão do Modeo ---------------------------------------------------------------

    entrada_modelo = {
        "model_year": widget_ano, 
        "transmission": widget_transmissao,  
        "fuel_type": widget_combustivel,  
        "city_l_100_km": widget_consumo_urbano,  
        "highway_l_100_km": widget_consumo_rodovia,  
        "combined_l_100_km": widget_consumo_combinado,  
        "vehicle_class_grouped": widget_classe_veiculo,  
        "engine_size_l_class": widget_tamanho_motor,  
        "cylinders_class": widget_cilindros,  
    }

    df_entrada_modelo = pd.DataFrame([entrada_modelo])

    if botao_previsao:
        previsao = modelo_xgb.predict(df_entrada_modelo)[0]
        st.metric(label='Emissao de CO2 prevista (g/km)', value=f"{previsao:.1f}")
