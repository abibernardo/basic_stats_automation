
import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd


resultados_categoricos = {}

def visualizar_medidas(df, c):
    try:
        if pd.api.types.is_numeric_dtype(df[c]):
            media = df[c].mean().round(3)
            moda = df[c].mode().values
            dp = df[c].std().round(3)
            min = df[c].min().round(3)
            max = df[c].max().round(3)
            assimetria = df[c].skew()
            # Calculando os quartis
            quartis = df[c].quantile([0.25, 0.5, 0.75]).round(3)
            quant_nan = st.session_state.df[c].isna().sum()
            st.write(f"### {c}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                        <div style="background-color: #008B8B; color: black; padding: 10px; border-radius: 5px; text-align: center;">
                            <strong>Média:</strong> {media}
                        </div>
                        <div style="background-color: #008B8B; color: black; padding: 10px; border-radius: 5px; text-align: center;">
                            <strong>Desvio Padrão:</strong> {dp}
                        </div>
                        <div style="background-color: #008B8B; color: black; padding: 10px; border-radius: 5px; text-align: center;">
                            <strong>Mínimo:</strong> {min}
                        </div>
                        <div style="background-color: #008B8B; color: black; padding: 10px; border-radius: 5px; text-align: center;">
                            <strong>Máximo:</strong> {max}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                                    <div style="background-color: #008B8B; color: black; padding: 10px; border-radius: 5px; text-align: center;">
                                        <strong>1º Quartil:</strong> {quartis.iloc[0]}
                                    </div>
                                    <div style="background-color: #008B8B; color: black; padding: 10px; border-radius: 5px; text-align: center;">
                                        <strong>Mediana:</strong> {quartis.iloc[1]}
                                    </div>
                                    <div style="background-color: #008B8B; color: black; padding: 10px; border-radius: 5px; text-align: center;">
                                        <strong>3º Quartil:</strong> {quartis.iloc[2]}
                                    </div>
                                    <div style="background-color: #008B8B; color: black; padding: 10px; border-radius: 5px; text-align: center;">
                                        <strong>Faltantes:</strong> {quant_nan}
                                    </div>
                                </div>
                                """,
                    unsafe_allow_html=True
                )
            st.divider()
            fig = px.histogram(df, x=c)
            st.plotly_chart(fig)
            st.divider()
        else:
            contagem = df[c].value_counts()
            val_unicos = df[c].nunique()
            moda = df[c].mode().values
            resultados_categoricos[c] = contagem
            col5, col6 = st.columns(2)
            with col5:
                st.write(contagem)
            with col6:
                category_counts = df[c].value_counts().reset_index()
                category_counts.columns = ['Categoria', 'Contagem']
                fig = px.pie(category_counts, names='Categoria', values='Contagem', title=' ')
                fig.update_traces(textinfo=None)
                st.plotly_chart(fig)
            st.write("### Contagem")
            st.bar_chart(contagem)
            st.divider()
    except Exception as ex:
        st.write("Ops, houve algum erro. Verifique a formatação da sua planilha")

@st.fragment
def visualizar_relacoes(df, x, y, cor, tamanho):
    if x not in '-' and y not in '-':
        if cor in '-' and tamanho in '-':
            st.scatter_chart(df, x=x, y=y)
        if cor not in '-' and tamanho in '-':
            fig = px.scatter(
                df,
                x=x,
                y=y,
                color=cor,
                color_continuous_scale="reds",
            )
            if fig:
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        if cor in '-' and tamanho not in '-':
            fig = px.scatter(
                df,
                x=x,
                y=y,
                size=tamanho,
                hover_data=None,
            )
            if fig:
                st.plotly_chart(fig, key="ia", on_select="rerun")
        if cor not in '-' and tamanho not in '-':
            fig = px.scatter(
                df,
                x=x,
                y=y,
                color=cor,
                size=tamanho,
                hover_data=None,
            )
            if fig:
                st.plotly_chart(fig, key="iis", on_select="rerun")
    else:
        st.write("Selecione as variáveis de interesse.")

@st.fragment
def barras(df, colunas_categoricas, colunas_numericas):
    col11, col22 = st.columns(2)
    with col11:
        default_ix = colunas_categoricas.index('-')
        variavel_numerica_bar = st.selectbox('Variável de interesse', colunas_numericas, key='s')
        variavel_divisora_bar = st.selectbox('Categoria dividora', colunas_categoricas, key='bar222')
    with col22:
        subcategoria = st.selectbox('Categoria divisora por cor (opcional)', colunas_categoricas, index = default_ix, key='bar22')
    if variavel_numerica_bar not in '-' and variavel_divisora_bar not in '-' and subcategoria in '-':
        fig = px.histogram(df, y=variavel_numerica_bar, x=variavel_divisora_bar)
        st.plotly_chart(fig)
    elif variavel_numerica_bar not in '-' and variavel_divisora_bar not in '-' and subcategoria not in '-':
        fig = px.histogram(
            df,
            x=variavel_divisora_bar,
            y=variavel_numerica_bar,
            color=subcategoria,  # Variável categórica para separação de cores
            barmode='group',  # Exibe barras agrupadas
        )
        st.plotly_chart(fig)




@st.fragment
def boxplots(df, colunas_categoricas, colunas_numericas):
    col11, col22 = st.columns(2)
    with col11:
        variavel_categorica_boxplot = st.selectbox('Categoria por boxplot', colunas_categoricas, key='key11')
        variavel_numerica_boxplot = st.selectbox('Variável de interesse', colunas_numericas)
    with col22:
        default_ix = colunas_categoricas.index('-')
        variavel_divisora_boxplot = st.selectbox('Categoria divisora por cor (opcional)', colunas_categoricas, index = default_ix, key='keyy22')
    if variavel_categorica_boxplot not in '-' and variavel_numerica_boxplot not in '-' and variavel_divisora_boxplot in '-':
        fig = px.box(df, x=variavel_categorica_boxplot, y=variavel_numerica_boxplot, title=" ")
        st.plotly_chart(fig)
    elif variavel_categorica_boxplot not in '-' and variavel_numerica_boxplot not in '-' and variavel_divisora_boxplot not in '-':
            fig = px.box(df, x=variavel_categorica_boxplot, y=variavel_numerica_boxplot,
                         color=variavel_divisora_boxplot)
            fig.update_traces(quartilemethod="linear")
            st.plotly_chart(fig)

@st.fragment
def linhas(df, colunas_categoricas, colunas_numericas):
    col11, col22 = st.columns(2)
    with col11:
        variavel_x = st.selectbox('Eixo x', colunas_numericas, key='hsbx11')
        variavel_y = st.selectbox('Eixo y', colunas_numericas, key='rsuua')
    with col22:
        default_ix = colunas_categoricas.index('-')
        variavel_cor = st.selectbox('Categoria divisora por cor (opcional)', colunas_categoricas,
                                                 index=default_ix, key='ifr4h')
    if variavel_x not in '-' and variavel_y not in '-' and variavel_cor not in '-':
        try:
            df1 = df.sort_values(by=variavel_x)
            fig = px.line(df1, x=variavel_x, y=variavel_y, color=variavel_cor)
            st.plotly_chart(fig)
        except Exception as e:
            st.write("Selecione variáveis válidas.")
    elif variavel_x not in '-' and variavel_y not in '-' and variavel_cor in '-':
        df1 = df.sort_values(by=variavel_x)
        fig = px.line(df1, x=variavel_x, y=variavel_y)
        st.plotly_chart(fig)


@st.fragment
def pairplor(df, variaveis_corr, cor_pp):
    df_numeric = df[variaveis_corr]
    corr = df_numeric.corr()
    if cor_pp not in '-':
        try:
            fig = px.scatter_matrix(df, dimensions=variaveis_corr, color=cor_pp)
            st.plotly_chart(fig)
        except Exception as e:
            st.write("Selecione variáveis válidas.")
    else:
        try:
            fig = px.scatter_matrix(df_numeric, dimensions=variaveis_corr)
            st.plotly_chart(fig)
        except Exception as e:
            st.write("Selecione variáveis válidas.")


def analise_descritiva(df):
    st.divider()
    column_names = df.columns
    colunas = column_names.tolist()
    colunas.append('-')
    st.write(f"## Análise Descritiva")
    col1, col2 = st.columns(2)
    with col1:
        option = st.selectbox(
            'Qual variável quer analisar?',
            column_names, key='descritiva')
    visualizar_medidas(df, option)

# @st.fragment
def analise_exploratoria(df):
    column_names = df.columns
    colunas = column_names.tolist()
    colunas.append('-')
    colunas_categoricas = df.select_dtypes(include=[object, 'category', 'datetime']).columns.tolist()
    colunas_numericas = df.select_dtypes(include=[np.number, 'datetime']).columns.tolist()
    colunas_numericas.append('-')
    colunas_categoricas.append('-')
    default_ix = colunas_numericas.index('-')
    default_ix2 = colunas_categoricas.index('-')
    default_ix3 = colunas_categoricas.index('-')
    grafico = st.radio("Que tipo de gráfico deseja construir?", ["Medidas", "Barras", "Dispersão", "Boxplot", "Linha", "Pares"], horizontal=True)
    if grafico == "Medidas":
        analise_descritiva(st.session_state.df)
    if grafico == "Barras":
        barras(df, colunas_categoricas, colunas_numericas)
        st.divider()
    if grafico == "Dispersão":
        col15, col25 = st.columns(2)
        with col15:
            x = st.selectbox('eixo x', colunas_numericas)
            y = st.selectbox('eixo y', colunas_numericas)
        with col25:
            cor = st.selectbox('Divisão por cor', colunas, index = default_ix3)
            tamanho = st.selectbox('Divisão por tamanho', colunas_numericas, index = default_ix)
            st.write("**Para deixar um campo vazio, selecione ' - '**")
        visualizar_relacoes(df, x, y, cor, tamanho)
        st.divider()
    if grafico == "Boxplot":
        boxplots(df, colunas_categoricas, colunas_numericas)
        st.divider()
    if grafico == "Linha":
        linhas(df, colunas_categoricas, colunas_numericas)
        st.divider()
    if grafico == "Pares":
        col9, col99 = st.columns(2)
        with col9:
            variaveis_corr = st.multiselect('Variáveis de interesse', colunas_numericas)
        with col99:
            cor_pp = st.selectbox('divisão por cor', colunas_categoricas, index=default_ix2)
            st.write("**Para deixar o campo vazio, selecione ' - '**")
        if st.button('Gerar!'):
            if '-' not in variaveis_corr:
                pairplor(df, variaveis_corr, cor_pp)
        st.divider()


st.title("📊 Análise Exploratória")

if 'df' in st.session_state:
    st.divider()
    analise_exploratoria(st.session_state.df)
else:
    st.info("Faça o upload de um dataset para começar.")


