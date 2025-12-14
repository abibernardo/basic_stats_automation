from pathlib import Path
import streamlit as st
import pandas as pd


# ---------- UPLOAD ----------
def present_file(uploaded_file):

    if "df" in st.session_state:
        return  # já carregado

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in {'.xls', '.xlsx'}:
        df = pd.read_excel(uploaded_file)
    elif suffix == '.csv':
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        st.error("Extensão não suportada")
        return

    st.session_state.df = df
    st.session_state.df_original = df.copy()

    st.success(f"✅ Dataset carregado ({df.shape[0]} × {df.shape[1]})")


# ---------- MANIPULAÇÃO ----------
def manipulacao_dados():

    df = st.session_state.df

    tipos_df = pd.DataFrame({
        "Variável": df.columns,
        "Tipo atual": df.dtypes.astype(str)
    })
    st.dataframe(tipos_df)

    col1, col2 = st.columns(2)

    with col1:
        coluna = st.selectbox("Selecione a variável", df.columns)

    with col2:
        novo_tipo = st.selectbox(
            "Novo tipo",
            ["Numérica", "Categórica", "Texto", "Data"]
        )

    if st.button("Aplicar alteração"):
        if novo_tipo == "Numérica":
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
        elif novo_tipo == "Categórica":
            df[coluna] = df[coluna].astype("category")
        elif novo_tipo == "Texto":
            df[coluna] = df[coluna].astype(str)
        elif novo_tipo == "Data":
            df[coluna] = pd.to_datetime(df[coluna], errors="coerce")

        st.session_state.df = df
        st.success(f"✅ {coluna} convertido para {novo_tipo}")



# ---------- APP ----------
st.title("📊 Tratamento de Dados")


if "df" in st.session_state:
    st.divider()
    manipulacao_dados()
else:
    st.info("Faça o upload de um dataset para começar.")
