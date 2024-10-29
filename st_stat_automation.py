import statsmodels.api as sm
import polars as pl
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
import streamlit as st
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import plotly.express as px
import statsmodels.api as sm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
import plotly.figure_factory as ff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
            st.write("**Gráfico de dispersão**")
            st.scatter_chart(df[c])
            st.divider()
            st.write("**Histograma**")
            fig = px.histogram(df, x=c)
            st.plotly_chart(fig)
            st.divider()
            st.write("**Gráfico de linha**")
            st.line_chart(df[c])
            st.divider()
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
def visualizar_relacoes(df, var_numericas):
    df = df[var_numericas]
    st.write("**Gráfico de dispersão**")
    st.scatter_chart(df)
    st.divider()
    st.write("**Histograma**")
    fig = px.histogram(df)
    st.plotly_chart(fig)
    st.divider()
    st.write("**Gráfico de linhas**")
    st.line_chart(df)

@st.fragment
def boxplots(df, colunas_categoricas, colunas_numericas):
    variavel_categorica_boxplot = st.selectbox('Categoria por boxplot', colunas_categoricas, key='key11')
    variavel_numerica_boxplot = st.selectbox('Variável de interesse', colunas_numericas)
    variavel_divisora_boxplot = st.selectbox('Categoria divisora por cor (opcional)', colunas_categoricas, key='keyy22')
    if st.button("Visualizar", key='boxp'):
        fig = px.box(df, x=variavel_categorica_boxplot, y=variavel_numerica_boxplot, title=" ")
        st.plotly_chart(fig)
        if variavel_categorica_boxplot and variavel_categorica_boxplot != variavel_divisora_boxplot:
            st.divider()
            fig = px.box(df, x=variavel_categorica_boxplot, y=variavel_numerica_boxplot,
                         color=variavel_divisora_boxplot)
            fig.update_traces(quartilemethod="linear")
            st.plotly_chart(fig)


@st.fragment
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

@st.fragment
def graficos_dispersao(df, x_dc, y_dc, cor, tamanho):
    st.divider()
    st.write(f"{x_dc} x {y_dc}")
    col20, col21 = st.columns(2)
    with col20:
        fig = px.scatter(df, x=x_dc, y=y_dc)
        st.plotly_chart(fig, key="simples", on_select="rerun")
    with col21:
        fig = px.scatter(df, x=x_dc, y=y_dc, trendline="ols")
        fig.update_traces(selector=dict(type='scatter', mode='lines'),
                          line=dict(color='red', width=4))
        st.plotly_chart(fig, key="simpless", on_select="rerun")
    st.divider()
    st.write(f"{x_dc} x {y_dc} dividido por {cor}")
    fig = px.scatter(
        df,
        x=x_dc,
        y=y_dc,
        color=cor,
        color_continuous_scale="reds",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.divider()
    st.write(f"{x_dc} x {y_dc} dividido por {cor}")
    fig = px.scatter(df, x=x_dc, y=y_dc, color=cor, marginal_x="rug", marginal_y="violin")
    st.plotly_chart(fig)
    fig = px.scatter(df, x=x_dc, y=y_dc, color = cor, marginal_y="histogram", marginal_x="box", trendline="ols", template="simple_white")
    st.plotly_chart(fig)
    st.divider()
    st.write(f"{x_dc} x {y_dc} por escala de cor em {tamanho}")
    fig = px.scatter(
        df,
        x=x_dc,
        y=y_dc,
        color=tamanho,
        color_continuous_scale="reds",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.divider()
    st.write(f"{x_dc} x {y_dc} dividido por cor em {cor} e tamanho em {tamanho}")
    fig = px.scatter(
        df,
        x=x_dc,
        y=y_dc,
        color=cor,
        size=tamanho,
        hover_data=None,
    )


    st.plotly_chart(fig, key="iris", on_select="rerun")


def analisar_residuos(modelo, X):
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
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.write("### Teste de Normalidade dos Resíduos (Shapiro-Wilk)")
        st.write(f"Estatística W: **{shapiro_test.statistic:.4f}**, p-valor: **{shapiro_test.pvalue:.4f}**")

        # fig_qq = sm.qqplot(residuos, line='45')
        # st.write("**QQ Plot**")
        # st.pyplot(fig_qq)
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

def preditos_x_reais(modelo, X, y):
    y_pred = modelo.predict(X)

    # Cria o gráfico de dispersão com Plotly
    fig = go.Figure()

    # Adiciona os pontos reais vs preditos
    fig.add_trace(go.Scatter(
        x=y,
        y=y_pred,
        mode="markers",
        name="Valores Preditos",
        marker=dict(color="blue")
    ))

    # Adiciona a linha de referência (y = x)
    fig.add_trace(go.Scatter(
        x=[y.min(), y.max()],
        y=[y.min(), y.max()],
        mode="lines",
        name="Linha de Referência (y=x)",
        line=dict(color="red", dash="dash")
    ))

    # Configurações do layout
    fig.update_layout(
        title="Gráfico de Valores Reais vs. Valores Preditos",
        xaxis_title="Valores Reais",
        yaxis_title="Valores Preditos",
        showlegend=True,
        width=800,
        height=600
    )

    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig)



@st.fragment
def novas_obs(modelo, X_col, y_col):
    if "novas_observacoes" not in st.session_state:
        st.session_state["novas_observacoes"] = pd.DataFrame(columns=modelo.model.exog_names + [y_col])
    if "prev_X_col" not in st.session_state or st.session_state["prev_X_col"] != X_col:
        st.session_state["novas_observacoes"] = pd.DataFrame(columns=modelo.model.exog_names + [y_col])
        st.session_state["prev_X_col"] = X_col
    novos_valores = {}
    # Coleta os novos valores do usuário
    for col in X_col:
        if st.session_state.df[col].dtype == 'object':
            novos_valores[col] = st.selectbox(f"{col}", st.session_state.df[col].unique())
        else:
            novos_valores[col] = st.number_input(f"{col}", value=0.0)

    # Quando o botão "Predição" é pressionado
    if st.button("Predição", key="regressao_lin"):
        # Cria um DataFrame a partir dos novos valores fornecidos
        nova_obs = pd.DataFrame([novos_valores])
        colunas_categoricas = nova_obs.select_dtypes(include=[object]).columns.tolist()
        nova_obs = pd.get_dummies(nova_obs, columns=colunas_categoricas, drop_first=True)
        nova_obs['const'] = 1
        nova_obs = nova_obs.reindex(columns=modelo.model.exog_names)
        # Realiza a predição
        predicao = modelo.get_prediction(nova_obs.astype(float))
        predicao_summary = predicao.summary_frame(alpha=0.05)  # 95% de intervalo de confiança
        predicao1 = modelo.predict(nova_obs)[0]
        nova_obs[y_col] = predicao1
        st.session_state["novas_observacoes"] = pd.concat([st.session_state["novas_observacoes"], nova_obs],
                                                          ignore_index=True)

        # Exibe a predição
        st.write(f"Previsão de {y_col}: {predicao_summary['mean'][0].round(3)}")
        st.write(
            f"Intervalo de Confiança (95%): [{predicao_summary['mean_ci_lower'][0].round(3)}, {predicao_summary['mean_ci_upper'][0].round(3)}]")
        # 95% de intervalo de confiança
        st.divider()
    st.write("## Predição das novas observações:")
    try:
        st.session_state["novas_observacoes"] = st.session_state["novas_observacoes"].drop(columns=["const"])
        st.dataframe(st.session_state["novas_observacoes"])
    except Exception as e:
        st.write(" ")


@st.fragment
def knn_pred(knn, X_col, y_col, scaler):
    if "novas_observacoes_knn" not in st.session_state:
        # Inicializa com as colunas de X_col e a coluna de predição
        st.session_state["novas_observacoes_knn"] = pd.DataFrame(columns=X_col + [y_col])

    # Verifica se o conjunto de preditores (X_col) mudou
    if "prev_X_col_knn" not in st.session_state or st.session_state["prev_X_col_knn"] != X_col:
        # Reinicializa o DataFrame de novas observações com as novas colunas
        st.session_state["novas_observacoes_knn"] = pd.DataFrame(columns=X_col + [y_col])
        st.session_state["prev_X_col_knn"] = X_col
    novos_valores_knn = {}
    # Coleta os novos valores do usuário
    for col in X_col:
        novos_valores_knn[col] = st.number_input(f"{col}", value=0.0)
    if st.button("Predição", key='knn_pred'):
        novos_valores_knn_df = pd.DataFrame([novos_valores_knn])
        novos_valores_knn_scaled = scaler.transform(
            novos_valores_knn_df)  # Apenas `transform` para manter a escala de treino
        predicao_nova_obs = knn.predict(novos_valores_knn_scaled)
        novos_valores_knn_df[y_col] = predicao_nova_obs
        st.session_state["novas_observacoes_knn"] = pd.concat([st.session_state["novas_observacoes_knn"], novos_valores_knn_df],
                                                          ignore_index=True)
    try:
        st.dataframe(st.session_state["novas_observacoes_knn"])
    except Exception as e:
        st.write(" ")



# @st.fragment
def analise_descritiva(df):
    st.divider()
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
    col1, col2 = st.columns(2)
    if visualisar == "Descrição das variáveis":
        with col1:
            st.divider()
            option = st.selectbox(
                'Qual variável quer analisar?',
                column_names, key='descritiva')
        visualizar_medidas(df, option)

    if visualisar == "Relação entre variáveis":
        colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write("## Análises de variáveis numéricas")
        var_numericas = st.multiselect('Variáveis numéricas de interesse', colunas_numericas)
        if st.button("Visualizar", key='numericas'):
            visualizar_relacoes(df, var_numericas)
        st.divider()

        st.write("## Boxplots por categoria")
        boxplots(df, colunas_categoricas, colunas_numericas)
        st.divider()

        st.write("## Matriz de correlação")
        variaveis_corr = st.multiselect(' ', colunas_numericas)

        if st.button("Visualizar", key='corr'):
            correlacao(df, variaveis_corr)
        st.divider()

        st.write("## Gráficos de dispersão mais avançados")

        x_dc = st.selectbox('Variável X', colunas_numericas)
        y_dc = st.selectbox('Variável Y', colunas_numericas)
        cor = st.selectbox('Variável categórica divisora:', colunas_categoricas)
        tamanho = st.selectbox('Variável numérica divisora:', colunas_numericas)
        st.divider()
        if st.button("Visualizar", key='disper'):
            graficos_dispersao(df, x_dc, y_dc, cor, tamanho)



def analise_regressao(df):
    st.title("Análise de Regressão Linear")
    column_names = st.session_state.df.columns
    colunas_numericas = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
    #colunas_categoricas = df.select_dtypes(include=[object]).columns.tolist()
    y_col = st.selectbox('Selecione a variável resposta (Y)', colunas_numericas)
    X_col = st.multiselect('Selecione os preditores (X)', column_names)

    if st.button("Modelar!", key='reglin'):
        if y_col and X_col:
            st.session_state.df = st.session_state.df.dropna()
            X_1 = st.session_state.df[X_col]
            colunas_categoricas = X_1.select_dtypes(include=[object]).columns.tolist()
            X = pd.get_dummies(X_1, columns=colunas_categoricas, drop_first=True)
            X = sm.add_constant(X)
            y = st.session_state.df[y_col]


            modelo = sm.OLS(y, X.astype(float)).fit()

            ### ANOVA
            formula = f"{y_col} ~ {' + '.join(X_col)}"
            modelo_anova = ols(formula, data=df).fit()
            anova_resultado = anova_lm(modelo_anova)


            stats_dict = {
                'Preditor': modelo.params.index,
                'Coeficiente': modelo.params.round(4).values,
                'Std Error': modelo.bse.round(4).values,
                'T-valor': modelo.tvalues.round(4).values,
                'P(>|t|)': modelo.pvalues.round(4).values
            }
            stats_df = pd.DataFrame(stats_dict)


            st.write("## Tabela de Coeficientes")
            st.table(stats_df)
            st.divider()
            st.write("## Tabela ANOVA")
            st.table(anova_resultado)
            st.divider()
            st.write(f"### Coeficiente de determinação", round(modelo.rsquared_adj, 4))

            col1, col2 = st.columns(2)
            i = 0
            for col in X_col:
                if i // 2 == 0:
                    with col1:
                        if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                            fig = px.scatter(df, x=col, y=y_col, trendline='ols', title=f'{col} vs {y_col}')
                            st.plotly_chart(fig)
                            i += 1
                else:
                    with col2:
                        if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                            fig = px.scatter(df, x=col, y=y_col, trendline='ols', title=f'{col} vs {y_col}')
                            st.plotly_chart(fig)
                            i += 1



            # Gráfico de resíduos
            st.write("**Atenção: modelos de regressão só são úteis se cumprirem pressupostos estatísticos que validam os seus resultados. Por isso, recomenda-se fortemente sempre analisar os resíduos do seu modelo.**")
            st.divider()
            st.write("# Análise de resíduos")
            analisar_residuos(modelo, X)
            st.write("# Predição")
            preditos_x_reais(modelo, X, y)
            st.divider()
            novas_obs(modelo, X_col, y_col)

def knn(df):
    st.title("K-Vizinhos mais Próximos")
    try:
        st.session_state.df = st.session_state.df.dropna()
        colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        y_col = st.selectbox('Selecione a variável resposta (Y)', colunas_categoricas)
        X_col = st.multiselect('Selecione os preditores (X)', colunas_numericas)

        if y_col and X_col:
            df = st.session_state.df[X_col + [y_col]].dropna(how='all')
            X = df.drop(y_col, axis=1)
            y = df[y_col]
            np.random.seed(42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Scale the features using StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Inicializar lista para armazenar o erro percentual
            perc_erro = []

            # Loop para testar diferentes valores de K
            for i in range(1, 30):
                # Criar o modelo KNN com k = i
                knn = KNeighborsClassifier(n_neighbors=i)

                # Treinar o modelo
                knn.fit(X_train_scaled, y_train)

                # Fazer previsões no conjunto de teste
                previsoes = knn.predict(X_test_scaled)

                # Calcular o percentual de erro
                perc_erro.append(np.mean(previsoes != y_test))

            # Criar DataFrame com valores de K e os erros percentuais
            k_values = np.arange(1, 30)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=k_values,
                y=perc_erro,
                mode='lines+markers',
                line=dict(dash='dot', color='red'),
                marker=dict(symbol='circle'),
                name='Percentual de Erro'
            ))
            fig.update_layout(
                title='Erro Percentual para diferentes valores de K',
                xaxis_title='K-vizinhos',
                yaxis_title='Percentual de Erro',
                template="plotly_white",
                width=800,
                height=600
            )

            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)
            st.divider()
            st.plotly_chart(fig)
            st.divider()


            # Treinar o modelo KNN com o número de vizinhos selecionado

            vizinhos = st.slider("Quantos k-vizinhos considerar?", 1, 30, 1)
            if st.button("Modelar!", key='knn'):
                knn = KNeighborsClassifier(n_neighbors=vizinhos)
                knn.fit(X_train_scaled, y_train)

                # Fazer previsões com o modelo ajustado
                y_pred = knn.predict(X_test_scaled)

                # Calcular a precisão
                accuracy = accuracy_score(y_test, y_pred)
                accuracy = round(accuracy, 3)
                st.divider()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f'### Previsão de "{y_col}"')
                with col2:
                    st.write(f'### {vizinhos} vizinhos mais próximos')
                with col3:
                    st.write(f'### Precisão de {accuracy}')
                st.divider()
                st.write(" ")
                conf_matrix = confusion_matrix(y_test, y_pred)
                with col2:
                    st.write(conf_matrix)
                st.write("# Predição")
                knn_pred(knn, X_col, y_col, scaler)
    except Exception as e:
        st.write(e)
        st.write("## OPS! Houve alguma inconsistência...")
        st.write(" ")
        st.write("Averigue se:")
        st.write("1. A variável RESPOSTA é categórica e está escrita corretamente")
        st.write("2. As outras colunas são numéricas OU codificadas numéricamente")
        st.write("3. Todas as colunas existem")
        st.write("4. Os valores faltantes foram devidamente preenchidos:")
        st.data_editor(df)

def present_excel(excel_path):
    df = None  # Inicializa df como None para evitar erro de escopo
    try:
        excel_file = pd.ExcelFile(excel_path)
        df = pd.read_excel(excel_file)
    except Exception as e:
        try:
            # Tentar carregar como CSV
            df = pd.read_csv(excel_path)
        except Exception as e:
            st.write("Erro ao carregar o arquivo. Certifique-se de que o formato está correto.")

    # Atribui o DataFrame ao session_state se foi carregado com sucesso
    if df is not None:
        st.session_state.df = df

    # Verifica se 'df' foi atribuído com sucesso antes de acessar as opções do ticker
    if 'df' in st.session_state:
        if ticker == "Análise exploratória":
            st.title("**Análise exploratória**")
            analise_descritiva(st.session_state.df)

        elif ticker == "Modelo de regressão":
            analise_regressao(st.session_state.df)

        elif ticker == "Modelo de classificação":
            knn(st.session_state.df)
    else:
        st.write("Não foi possível carregar os dados para continuar.")


resultados_categoricos = {}

st.sidebar.subheader("Escolha a análise que deseja realizar")

secs = ["Análise exploratória", "Modelo de regressão", "Modelo de classificação"]
tickers = secs
ticker = st.sidebar.selectbox("Seções", tickers)


excel_path = st.file_uploader("Escolha um conjunto de dados para analisar", type=["xlsx", "xls", "csv"])
if excel_path is not None:
    present_excel(excel_path)


if st.session_state.get('filtros_aplicados', []):
    st.sidebar.write(f"**Filtros aplicados:** {', '.join(st.session_state.get('filtros_aplicados', []))}")
