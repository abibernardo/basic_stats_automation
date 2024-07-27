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

def manipulacao_de_dados(df):
    st.data_editor(df)
    column_names = st.session_state.df.columns
    tratamentos = [":)", "Valores faltantes", "Tipos", "Filtragem de dados"]
    tratamento = st.selectbox(
        'Qual tratamento deseja fazer?',
        tratamentos)

    if tratamento == "Tipos":
        column = st.selectbox(
            'Quer verificar o tipo de qual coluna?',
            column_names)
        tipo = st.session_state.df[column].dtypes
        st.write(f"**{column} é do tipo {tipo}**")
        st.write(f"Caso o tipo da variável esteja inadequado, para que tipo deseja alterar a coluna {column}?")
        tipos = ["-", "string", "numerical", "date"]
        novo_tipo = st.selectbox(
            'Escolha o tipo adequado',
            tipos)
        if st.button("Mudar tipo da variável"):
            try:
                if novo_tipo == 'string':
                    st.session_state.df[column] = st.session_state.df[column].astype(str)
                elif novo_tipo == 'numerical':
                    st.session_state.df[column] = pd.to_numeric(st.session_state.df[column], errors='raise')
                elif novo_tipo == 'date':
                    st.session_state.df[column] = pd.to_datetime(st.session_state.df[column], errors='raise')
                st.write(f"**Tipo da coluna {column} alterado para {novo_tipo}!**")
            except Exception as e:
                st.write(f"Erro ao tentar alterar o tipo da coluna: {e}")

    if tratamento == "Valores faltantes":
        column = st.selectbox(
            'Quer verificar valores faltantes de qual coluna?',
            column_names)
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
            st.write(f"**Deseja excluir as colunas com dados faltantes em '{column}'?**")
            if st.button("Excluir linhas NA"):
                st.session_state.df = st.session_state.df.dropna(subset=[column])
                st.write(f"**Linhas com dados faltantes em '{column}' excluídas!**")


    if tratamento == "Filtragem de dados":
        filtragem = st.selectbox(
            'Por qual variável quer filtrar?',
            column_names)
        filtros_aplicados = st.session_state.get('filtros_aplicados', [])

        if pd.api.types.is_numeric_dtype(st.session_state.df[filtragem]):
            valor = st.session_state.df[filtragem].unique()
            valor_coluna = st.multiselect(
                "Por quais valores deseja filtrar?",
                valor)
            if st.button("Aplicar filtro"):
                # Armazene uma cópia original do DataFrame, se ainda não tiver feito isso
                if 'df_copy' not in st.session_state:
                    st.session_state.df_copy = st.session_state.df.copy()
                st.session_state.df = st.session_state.df[st.session_state.df[filtragem].isin(valor_coluna)]
                st.write(st.session_state.df)
                st.write(
                    "Para desativar filtros, selecione a variável filtrada e clique em **'resetar filtro'**")
                # Adiciona a variável filtrada na lista de filtros aplicados
                if filtragem not in filtros_aplicados:
                    filtros_aplicados.append(filtragem)
                st.session_state.filtros_aplicados = filtros_aplicados
            if st.button("Resetar filtro"):
                if 'df_copy' in st.session_state:
                    st.session_state.df = st.session_state.df_copy.copy()
                    st.write("DataFrame original:")
                    st.write(st.session_state.df)
                    filtros_aplicados.clear()

        else:
            valor = st.session_state.df[filtragem].unique()
            valor_coluna = st.multiselect(
                "Por quais valores deseja filtrar?",
                valor)
            if st.button("Aplicar filtro"):
                # Armazene uma cópia original do DataFrame, se ainda não tiver feito isso
                if 'df_copy' not in st.session_state:
                    st.session_state.df_copy = st.session_state.df.copy()
                st.session_state.df = st.session_state.df[st.session_state.df[filtragem].isin(valor_coluna)]
                st.write(st.session_state.df)
                st.write(
                    "Para desativar filtros, selecione a variável filtrada e clique em **'resetar filtro'**")
                # Adiciona a variável filtrada na lista de filtros aplicados
                if filtragem not in filtros_aplicados:
                    filtros_aplicados.append(filtragem)
                st.session_state.filtros_aplicados = filtros_aplicados



            # Botão para desfazer a filtragem e voltar ao DataFrame original
            if st.button("Resetar filtro"):
                if 'df_copy' in st.session_state:
                    st.session_state.df = st.session_state.df_copy.copy()
                    st.write("DataFrame original:")
                    st.write(st.session_state.df)
                    filtros_aplicados.clear()


def analise_descritiva(df):
    st.markdown(f"___________________________________________________________________</p>", unsafe_allow_html=True)
    st.dataframe(df, width=900)
    visualisar = st.selectbox(
        'O que quer visualizar?',
        ["-", "Descrição das variáveis", "Relação entre variáveis"])
    column_names = df.columns
    if 'graph_choices' not in st.session_state:
        st.session_state.graph_choices = {}
    if visualisar == "Descrição das variáveis":
        option = st.multiselect(
            'Quais variáveis quer analisar?',
            column_names, key='unique_key_1')
        graphs = ["scatter", "histograma", "line", "boxplot"]
        graph = st.multiselect(
            f'Que gráficos você deseja usar para variáveis numéricas?',
            graphs)
        if st.button("Visualizar"):
            for idx, c in enumerate(option):
                try:
                    st.write("_____________________________________")
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
                        st.write("Média", media)
                        st.write("Mínimo", min)
                        st.write("Máximo", max)
                        st.write("Desvio Padrão", dp)
                        st.write("1º Quartil", quartis.iloc[0])
                        st.write("2º Quartil (Mediana)", quartis.iloc[1])
                        st.write("3º Quartil", quartis.iloc[2])
                        if quant_nan:
                            st.write(quant_nan, "dados faltantes")

                        if "scatter" in graph:
                            st.scatter_chart(df[c])
                        if "histograma" in graph:
                            fig = px.histogram(df, x=c, title=f'Histograma de {c}')
                            st.plotly_chart(fig)
                        if "line" in graph:
                            st.line_chart(df[c])
                        if "boxplot" in graph:
                            fig = px.box(df, y=c, title=f'Boxplot de {c}')
                            st.plotly_chart(fig)
                    else:
                        contagem = df[c].value_counts()
                        val_unicos = df[c].nunique()
                        moda = df[c].mode().values
                        resultados_categoricos[c] = contagem
                        st.write(f"## Análise descritiva da variável {c}")
                        st.write(contagem)
                        st.write(f'{val_unicos} valores únicos')
                        st.write(f'Moda = {val_unicos}')

                        st.bar_chart(contagem)
                    st.write("_____________________________________")
                except Exception as ex:
                    st.write("Ops, houve algum erro. Verifique a formatação da sua planilha")

    if visualisar == "Relação entre variáveis":
        colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        #variavel_categorica = st.multiselect('Categoria por boxplot', colunas_categoricas)
        #variavel_numerica = st.multiselect('Variável de interesse', colunas_numericas)
        st.write("## Análises de variáveis numéricas")
        var_numericas = st.multiselect('Variáveis numéricas de interesse', colunas_numericas)
        if st.button("Visualizar gráficos"):
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
            st.write("**Pair Plot**")
            fig = sns.pairplot(df)
            st.pyplot(fig)
            st.write("_____________________________________")
            st.write("**Gráfico de linhas**")
            st.line_chart(df)
        st.write("______________________________________")
        st.write("______________________________________")

        st.write("## Boxplots por categoria")
        variavel_categorica_boxplot = st.selectbox('Categoria por boxplot', colunas_categoricas)
        variavel_numerica_boxplot = st.selectbox('Variável de interesse', colunas_numericas)

        if st.button("Visualizar Boxplots"):
            fig = px.box(df, x=variavel_categorica_boxplot, y=variavel_numerica_boxplot, title=f'{variavel_numerica_boxplot} por {variavel_categorica_boxplot}')
            st.plotly_chart(fig)

        st.write("______________________________________")
        st.write("______________________________________")

        st.write("## Matriz de correlação")

        variaveis_corr = st.multiselect('Matriz de correlação', colunas_numericas)
        if st.button("Matriz de correlação"):
            df_numeric = df[variaveis_corr]
            corr = df_numeric.corr()
            fig = px.imshow(corr, text_auto=True, title='Matriz de Correlação')
            st.plotly_chart(fig)

        st.write("______________________________________")
        st.write("______________________________________")

        st.write("## Dispersão por cor (legenda selecionável)")
        cor = st.selectbox('Dividir por cor:', colunas_categoricas)
        tamanho = st.selectbox('Dividir por tamanho:', colunas_numericas)
        x_dc = st.selectbox('Variável X', colunas_numericas)
        y_dc = st.selectbox('Variável Y', colunas_numericas)
        if st.button("Mostrar gráfico"):
            fig = px.scatter(
                df,
                x=x_dc,
                y=y_dc,
                color=cor,
                size=tamanho,
                hover_data=None,
            )

            event = st.plotly_chart(fig, key="iris", on_select="rerun")

            event.selection




# Função para exibir a matriz de correlação

def inferencia(df):
    column_names = df.columns
    st.write("____________________________")

def analise_regressao(df):
    st.title("Análise de Regressão Linear")
    column_names = df.columns
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    #colunas_categoricas = df.select_dtypes(include=[object]).columns.tolist()
    y_col = st.selectbox('Selecione a variável resposta (Y)', colunas_numericas)
    X_col = st.multiselect('Selecione os preditores (X)', column_names)

    if st.button("Executar Regressão Linear"):
        if y_col and X_col:
            X = df[X_col]
            colunas_categoricas = X.select_dtypes(include=[object]).columns.tolist()
            X = pd.get_dummies(X, columns=colunas_categoricas, drop_first=True)
            X = sm.add_constant(X)

            y = df[y_col]


            #modelo = sm.OLS(y, X).fit()
            modelo = sm.OLS(y, X.astype(float)).fit()

            st.write(modelo.summary())

            # Visualização da regressão
            for col in X_col:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.scatter(df, x=col, y=y_col, trendline='ols', title=f'{col} vs {y_col}')
                    st.plotly_chart(fig)

            # Gráfico de resíduos
            st.write("**Atenção: modelos de regressão só são úteis se cumprirem pressupostos estatísticos que validam os seus resultados. Por isso, recomenda-se fortemente sempre analisar os resíduos do seu modelo.**")

            residuos = modelo.resid
            # Grafico residuos
            fig_residuos = go.Figure()
            fig_residuos.add_trace(go.Scatter(x=modelo.fittedvalues, y=residuos, mode='markers'))
            fig_residuos.add_trace(go.Scatter(x=modelo.fittedvalues, y=[0] * len(residuos), mode='lines'))
            fig_residuos.update_layout(title="Gráfico de Resíduos", xaxis_title="Valores Ajustados",
                                       yaxis_title="Resíduos")
            st.plotly_chart(fig_residuos)
            # Teste de Normalidade dos Resíduos (Shapiro-Wilk)
            shapiro_test = shapiro(residuos)
            st.write("**Teste de Normalidade dos Resíduos (Shapiro-Wilk)**")
            st.write(f"Estatística W: {shapiro_test.statistic:.4f}, p-valor: {shapiro_test.pvalue:.4f}")
            fig_qq = sm.qqplot(residuos, line='45')
            #st.write("**QQ Plot**")
            #st.pyplot(fig_qq)
            if shapiro_test.pvalue < 0.06:
                st.write("Rejeita-se hipótese de normalidade do resíduo **(!)**")
            else:
                st.write("Não rejeita-se hipótese de normalidade dos resíduos")

            # Teste de Homocedasticidade (Breusch-Pagan)
            _, bp_pvalue, _, _ = het_breuschpagan(residuos, X)
            st.write("**Teste de Homocedasticidade (Breusch-Pagan)**")
            st.write(f"p-valor: {bp_pvalue:.4f}")
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
        youtube_url = 'https://www.youtube.com/watch?v=2b5uvhE3-yw'
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

    if ticker == "Inferência":
        st.title("**Inferência**")
        inferencia(st.session_state.df)

    if ticker == "Modelo de regressão":
        analise_regressao(df)

resultados_categoricos = {}

st.title("ESTATÍSTICA BÁSICA SEMI AUTOMATIZADA")
st.markdown("**Análises estatísticas feitas de forma simples!**")


excel_path = st.file_uploader("Escolha um banco de dados para analisar", type=["xlsx", "xls"])

if excel_path is not None:
    present_excel(excel_path)
if st.session_state.get('filtros_aplicados', []):
    st.sidebar.write(f"**Filtros aplicados:** {', '.join(st.session_state.get('filtros_aplicados', []))}")



#streamlit run main.py
