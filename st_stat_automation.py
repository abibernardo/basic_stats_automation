import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import plotly.express as px
import statsmodels.api as sm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
import plotly.figure_factory as ff


def alterar_tipo(column):
    tipo = st.session_state.df[column].dtypes
    st.write(f"**{column} é do tipo {tipo}**")
    st.write("______________________________________________")
    st.write(f"### Caso o tipo da variável esteja inadequado, para que tipo deseja alterar a coluna {column}?")
    tipos = ["-", "categórica", "numérica", "data"]
    novo_tipo = st.selectbox(
        'Escolha o tipo adequado',
        tipos)
    if st.button("Mudar tipo da variável"):
        try:
            if novo_tipo == 'categórica':
                st.session_state.df[column] = st.session_state.df[column].astype(str)
            elif novo_tipo == 'numérica':
                st.session_state.df[column] = pd.to_numeric(st.session_state.df[column], errors='raise')
            elif novo_tipo == 'data':
                st.session_state.df[column] = pd.to_datetime(st.session_state.df[column], errors='raise')
            st.write(f"**Coluna {column} alterada para {novo_tipo}!**")
        except Exception as e:
            st.write(f"Erro ao tentar alterar o tipo da coluna: {e}")


def tratar_nan(column):
    nan_count = st.session_state.df[column].isna().sum()
    st.write(f"Há {nan_count} valores faltantes na coluna {column}.")
    if nan_count > 0 and pd.api.types.is_numeric_dtype(st.session_state.df[column]):
        fillna_modos = {"Excluir linha", "Substituir por zero", "Substituir pela média"}
        fill_na = st.selectbox(
            'Quer usar qual técnica de preenchimento?',
            fillna_modos)
        if st.button("Aplicar método de preenchimento"):
            if fill_na == "Excluir linha":
                st.session_state.df = st.session_state.df.dropna(subset=[column])
            if fill_na == "Substituir por zero":
                st.session_state.df[column] = st.session_state.df[column].fillna(0)
            if fill_na == "Substituir pela média":
                imputer = SimpleImputer(strategy='mean')
                st.session_state.df[column] = imputer.fit_transform(st.session_state.df[[column]])
            st.write(f"**Dados faltantes de '{column}' preenchidos com o método '{fill_na}'!**")
    elif nan_count > 0:
        st.write(f"**Deseja excluir as linhas com dados faltantes em '{column}'?**")
        if st.button("Excluir linhas NA"):
            st.session_state.df = st.session_state.df.dropna(subset=[column])
            st.write(f"**Linhas com dados faltantes em '{column}' excluídas!**")



def limpar_filtros(filtros_aplicados):
    if 'df_copy' in st.session_state:
        st.session_state.df = st.session_state.df_copy.copy()
        st.write("**Filtragem desfeita**")
        st.write(st.session_state.df)
        filtros_aplicados.clear()


def visualizar_medidas(df, option):
    for idx, c in enumerate(option):
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
                st.write(f"## Análise descritiva da variável {c}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Média", media)
                    st.write("Desvio Padrão", dp)
                    st.write("Mínimo", min)
                    st.write("Máximo", max)
                with col2:

                    st.write("1º Quartil", quartis.iloc[0])
                    st.write("2º Quartil (Mediana)", quartis.iloc[1])
                    st.write("3º Quartil", quartis.iloc[2])
                if quant_nan:
                    st.write(quant_nan, "dados faltantes")

                st.write("_____________________________________")
                st.write("**Gráfico de dispersão**")
                st.scatter_chart(df[c])
                st.write("_____________________________________")
                st.write("**Histograma**")
                fig = px.histogram(df, x=c)
                st.plotly_chart(fig)
                st.write("_____________________________________")
                st.write("**Gráfico de linha**")
                st.line_chart(df[c])
                st.write("_____________________________________")
                st.write("**Boxplot**")
                fig = px.box(df, y=c)
                st.plotly_chart(fig)
            else:
                contagem = df[c].value_counts()
                val_unicos = df[c].nunique()
                moda = df[c].mode().values
                resultados_categoricos[c] = contagem
                st.write(f"## Análise descritiva da variável {c}")
                col5, col6 = st.columns(2)
                with col5:
                    st.write(contagem)
                    st.write(f'Valores únicos: {val_unicos}')
                    st.write(f'Moda: {moda}')
                with col6:
                    category_counts = df[c].value_counts().reset_index()
                    category_counts.columns = ['Categoria', 'Contagem']
                    fig = px.pie(category_counts, names='Categoria', values='Contagem', title=' ')
                    fig.update_traces(textinfo=None)
                    st.plotly_chart(fig)
                st.write("### Contagem")
                st.bar_chart(contagem)




            st.write("_____________________________________")
        except Exception as ex:
            st.write("Ops, houve algum erro. Verifique a formatação da sua planilha")

def visualizar_relacoes(df, var_numericas):
    df = df[var_numericas]
    st.write("**Gráfico de dispersão**")
    st.scatter_chart(df)
    st.write("_____________________________________")
    st.write("**Histograma**")
    fig = px.histogram(df)
    st.plotly_chart(fig)
    group_labels = df.columns.tolist()  # Obter rótulos das colunas
    hist_data = [df[col].dropna().tolist() for col in df.columns]
    fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])
    st.plotly_chart(fig)
    st.write("_____________________________________")
    st.write("**Gráfico de linhas**")
    st.line_chart(df)

def correlacao(df, variaveis_corr):
    df_numeric = df[variaveis_corr]
    corr = df_numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=True,  # Exibe os valores das correlações automaticamente
        aspect="auto",  # Ajusta o aspecto para que cada célula seja quadrada
        color_continuous_scale=px.colors.diverging.RdBu,  # Paleta de cores divergente
        color_continuous_midpoint=0,  # Centraliza as cores em torno de 0
        title='Matriz de Correlação'  # Título do gráfico
    )

    # Ajustar o tamanho da figura
    fig.update_layout(
        width=800,  # Largura do gráfico
        height=600,  # Altura do gráfico
        margin=dict(l=60, r=60, t=60, b=60)  # Margens para ajustar espaçamento
    )

    # Atualizar os eixos para melhorar a leitura
    fig.update_xaxes(side="bottom", tickangle=-45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(fig)
    st.write("**Pair Plot**")
    fig = px.scatter_matrix(df_numeric)
    st.plotly_chart(fig)

def graficos_dispersao(df, x_dc, y_dc, cor, tamanho):
    st.write(f"______________________________")
    st.write(f"### {x_dc} x {y_dc}")
    fig = px.scatter(df, x=x_dc, y=y_dc)
    st.plotly_chart(fig, key="simples", on_select="rerun")
    st.write("______________________________________")
    st.write(f"### {x_dc} x {y_dc} dividido por {cor} (legenda selecionável)")
    fig = px.scatter(
        df,
        x=x_dc,
        y=y_dc,
        color=cor,
        color_continuous_scale="reds",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.write("______________________________________")
    st.write(f"### {x_dc} x {y_dc} dividido por {cor}")
    fig = px.scatter(df, x=x_dc, y=y_dc, color = cor, marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white")
    st.plotly_chart(fig)
    st.write("______________________________________")
    st.write(f"### {x_dc} x {y_dc} por escala de cor em {tamanho}")
    fig = px.scatter(
        df,
        x=x_dc,
        y=y_dc,
        color=tamanho,
        color_continuous_scale="reds",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.write("______________________________________")
    st.write(f"### {x_dc} x {y_dc} dividido por cor em {cor} e tamanho em {tamanho}")
    fig = px.scatter(
        df,
        x=x_dc,
        y=y_dc,
        color=cor,
        size=tamanho,
        hover_data=None,
    )

    st.plotly_chart(fig, key="iris", on_select="rerun")






def manipulacao_de_dados(df):
    st.data_editor(df)
    st.write("______________________________________________")
    column_names = st.session_state.df.columns
    tratamentos = ["Valores faltantes", "Alterar tipos", "Filtragem de dados"]
    st.write("## Deseja fazer que tipo de tratamento?")
    tratamento = st.radio(
        " ",
        tratamentos,
    )
    st.write("______________________________________________")

    if tratamento == "Alterar tipos":
        column = st.selectbox(
            'Quer verificar o tipo de qual coluna?',
            column_names)
        alterar_tipo(column)

    if tratamento == "Valores faltantes":
        column = st.selectbox(
            'Quer verificar valores faltantes de qual coluna?',
            column_names)
        tratar_nan(column)

    if tratamento == "Filtragem de dados":
        filtragem = st.selectbox(
            'Por qual variável quer filtrar?',
            column_names)
        filtros_aplicados = st.session_state.get('filtros_aplicados', [])

        valor = st.session_state.df[filtragem].unique()
        valor_coluna = st.multiselect("Por quais valores deseja filtrar?", valor)
        if st.button("Aplicar filtro", key='botao1'):
        # Armazene uma cópia original do DataFrame, se ainda não tiver feito isso
            if 'df_copy' not in st.session_state:
                st.session_state.df_copy = st.session_state.df.copy()
            st.session_state.df = st.session_state.df[st.session_state.df[filtragem].isin(valor_coluna)]
            st.write(st.session_state.df)
            st.write("Para desativar filtros, clique em **'resetar filtro'**")
            # Adiciona a variável filtrada na lista de filtros aplicados
            if filtragem not in filtros_aplicados:
                filtros_aplicados.append(filtragem)
            st.session_state.filtros_aplicados = filtros_aplicados
        if len(filtros_aplicados) > 0:
            if st.button("Resetar filtro", key='botao2'):
                limpar_filtros(filtros_aplicados)

# @st.fragment
def analise_descritiva(df):
    st.markdown(f"___________________________________________________________________</p>", unsafe_allow_html=True)
    st.dataframe(df, width=900)
    st.write("### O que quer visualizar?")
    visualisar = st.radio(
        " ",
        ["Descrição das variáveis", "Relação entre variáveis"],
        captions=[
            "Medidas estatísticas e distribuição",
            "Gráficos relacionando variáveis",
        ],
    )
    column_names = df.columns
    if visualisar == "Descrição das variáveis":
        st.write("_________________________________")
        st.write("## Quais variáveis quer analisar?")
        option = st.multiselect(
            ' ',
            column_names, key='unique_key_1')
        if st.button("Visualizar"):
            visualizar_medidas(df, option)

    if visualisar == "Relação entre variáveis":
        colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write("## Análises de variáveis numéricas")
        var_numericas = st.multiselect('Variáveis numéricas de interesse', colunas_numericas)
        if st.button("Visualizar gráficos"):
            visualizar_relacoes(df, var_numericas)
        st.write("______________________________________")
        st.write("______________________________________")

        st.write("## Boxplots por categoria")
        variavel_categorica_boxplot = st.selectbox('Categoria por boxplot', colunas_categoricas, key='key11')
        variavel_numerica_boxplot = st.selectbox('Variável de interesse', colunas_numericas)
        variavel_divisora_boxplot = st.selectbox('Categoria divisora por cor (opcional)', colunas_categoricas, key='keyy22')

        if st.button("Visualizar Boxplots"):
            fig = px.box(df, x=variavel_categorica_boxplot, y=variavel_numerica_boxplot, title=f'{variavel_numerica_boxplot} por {variavel_categorica_boxplot}')
            st.plotly_chart(fig)
            if variavel_categorica_boxplot != variavel_divisora_boxplot:
                fig = px.bar(df, x=variavel_categorica_boxplot, y=variavel_numerica_boxplot, color=variavel_divisora_boxplot, barmode="group")
                st.plotly_chart(fig)
        st.write("______________________________________")
        st.write("______________________________________")

        st.write("## Matriz de correlação")
        variaveis_corr = st.multiselect(' ', colunas_numericas)

        if st.button("Visualizar"):
            correlacao(df, variaveis_corr)
        st.write("______________________________________")
        st.write("______________________________________")

        st.write("## Gráficos de dispersão mais avançados")

        x_dc = st.selectbox('Variável X', colunas_numericas)
        y_dc = st.selectbox('Variável Y', colunas_numericas)
        cor = st.selectbox('Variável categórica divisora:', colunas_categoricas)
        tamanho = st.selectbox('Variável numérica divisora:', colunas_numericas)
        st.write("______________________________________")
        if st.button("Mostrar gráficos"):
            graficos_dispersao(df, x_dc, y_dc, cor, tamanho)



def analise_regressao(df):
    st.title("Análise de Regressão Linear")
    column_names = st.session_state.df.columns
    colunas_numericas = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
    #colunas_categoricas = df.select_dtypes(include=[object]).columns.tolist()
    y_col = st.selectbox('Selecione a variável resposta (Y)', colunas_numericas)
    X_col = st.multiselect('Selecione os preditores (X)', column_names)

    if st.button("Executar Regressão Linear"):
        if y_col and X_col:
            X = st.session_state.df[X_col]
            colunas_categoricas = X.select_dtypes(include=[object]).columns.tolist()
            X = pd.get_dummies(X, columns=colunas_categoricas, drop_first=True)
            X = sm.add_constant(X)

            y = st.session_state.df[y_col]


            modelo = sm.OLS(y, X.astype(float)).fit()

            stats_dict = {
                'Preditor': modelo.params.index,
                'Coeficiente': modelo.params.round(4).values,
                'Std Error': modelo.bse.round(4).values,
                'T-valor': modelo.tvalues.round(4).values,
                'P(>|t|)': modelo.pvalues.round(4).values
            }
            stats_df = pd.DataFrame(stats_dict)


            st.write("## Resultados do modelo")
            st.table(stats_df)
            st.write("_______________")
            st.write(f"### Coeficiente de determinação", round(modelo.rsquared_adj, 4))



            # Visualização da regressão
            col1, col2 = st.columns(2)
            i = 0
            for col in X_col:
                if i // 2 == 0:
                    with col1:
                        if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                            fig = px.scatter(df, x=col, y=y_col, trendline='ols', title=f'{col} vs {y_col}')
                            st.plotly_chart(fig)
                            i += 1
                            st.write("______________________________________")
                else:
                    with col2:
                        if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                            fig = px.scatter(df, x=col, y=y_col, trendline='ols', title=f'{col} vs {y_col}')
                            st.plotly_chart(fig)
                            i += 1
                            st.write("______________________________________")


            # Gráfico de resíduos
            st.write("**Atenção: modelos de regressão só são úteis se cumprirem pressupostos estatísticos que validam os seus resultados. Por isso, recomenda-se fortemente sempre analisar os resíduos do seu modelo.**")
            st.write("______________________________________")
            st.write("# Análise de resíduos")
            residuos = modelo.resid
            # Grafico residuos
            fig_residuos = go.Figure()
            fig_residuos.add_trace(go.Scatter(x=modelo.fittedvalues, y=residuos, mode='markers'))
            fig_residuos.add_trace(go.Scatter(x=modelo.fittedvalues, y=[0] * len(residuos), mode='lines'))
            fig_residuos.update_layout(title=" ", xaxis_title="Valores Ajustados",
                                           yaxis_title="Resíduos")
            st.plotly_chart(fig_residuos)
            # Teste de Normalidade dos Resíduos (Shapiro-Wilk)
            shapiro_test = shapiro(residuos)
            st.write("______________________________________")
            col3, col4 = st.columns(2)
            with col3:
                st.write("### Teste de Normalidade dos Resíduos (Shapiro-Wilk)")
                st.write(f"Estatística W: **{shapiro_test.statistic:.4f}**, p-valor: **{shapiro_test.pvalue:.4f}**")

                #fig_qq = sm.qqplot(residuos, line='45')
                #st.write("**QQ Plot**")
                #st.pyplot(fig_qq)
                if shapiro_test.pvalue < 0.06:
                    st.write("Rejeita-se hipótese de normalidade do resíduo **(!)**")
                else:
                    st.write("Não rejeita-se hipótese de normalidade dos resíduos")
            with col4:
                # Teste de Homocedasticidade (Breusch-Pagan)
                _, bp_pvalue, _, _ = het_breuschpagan(residuos, X)
                st.write("### Teste de Homocedasticidade (Breusch-Pagan)")
                st.write(f"p-valor: **{bp_pvalue:.4f}**")
                if bp_pvalue < 0.06:
                    st.write("Há indícios de heterocedasticidade no resíduo **(!)**")
                else:
                    st.write("Não rejeita-se hipótese de homocedasticidade do resíduo")



def present_excel(excel_path):
    excel_file = pd.ExcelFile(excel_path)
    df = pd.read_excel(excel_file)
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_excel(excel_file)

    st.sidebar.subheader("Escolha a análise que deseja realizar")

    secs = ["Apresentação", "Manipulação de dados", "Análise descritiva", "Modelo de regressão"]
    tickers = secs
    ticker = st.sidebar.selectbox("Seções", tickers)

    if ticker == "Apresentação":
        st.title("Olá!!!")

        st.markdown("""
        Esta aplicação foi criada com o propósito de simplificar análises estatísticas básicas, de modo a auxiliar que estudantes/pesquisadores das ciências humanas que não possuem familiaridade com linguagens de programação, possam visualizar seus dados sem terem que recorrer a softwares.
        """)

        st.header("Orientações:")
        st.markdown("""
        Antes de ir para as seções de análises estatísticas, faça o tratamento da sua planilha na seção **"Manipulação de dados"**. Lá, você poderá fazer as devidas alterações para que todas as outras etapas ocorram sem erros. Caso a aplicação dê resultados incoerentes, retorne a essa etapa para verificar aonde está o problema.
        *Por exemplo: se receber um gráfico que não faça sentido, verifique se sua variável é do tipo adequado no item **"Tipo"**, dentro de **"Manipulação de dados"**.*
        """)

        st.markdown("""
        Abaixo, deixo um vídeo-tutorial das funcionalidades da aplicação até o momento.
        """)
        youtube_url = 'https://youtu.be/YRx7nFkqafo'
        st.video(youtube_url)

        st.markdown("""
        Em caso de dúvidas, contate: [bernardoabib1@gmail.com](mailto:bernardoabib1@gmail.com)
        """)

    if ticker == "Manipulação de dados":
        st.title("**Manipulação de dados**")
        manipulacao_de_dados(st.session_state.df)

    if ticker == "Análise descritiva":
        st.title("**Análise descritiva**")
        analise_descritiva(st.session_state.df)

    if ticker == "Modelo de regressão":
        analise_regressao(df)

resultados_categoricos = {}

st.title("ESTATÍSTICA BÁSICA SEMI AUTOMATIZADA")
st.markdown("**Análises estatísticas feitas de forma simples!**")


excel_path = st.file_uploader("Escolha um banco de dados para analisar", type=["xlsx", "xls"])
url = 'https://raw.githubusercontent.com/abibernardo/basic_stats_automation/main/dados_bsa.xlsx'
st.write("Caso queira testar as funcionalidades, baixe um excel para testar:")
st.link_button('download', url)


if excel_path is not None:
    present_excel(excel_path)


if st.session_state.get('filtros_aplicados', []):
    st.sidebar.write(f"**Filtros aplicados:** {', '.join(st.session_state.get('filtros_aplicados', []))}")



#streamlit run main.py
