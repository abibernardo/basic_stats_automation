import polars as pl
from pathlib import Path
import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns



def present_file(uploaded_file):
    """
    Carrega um DataFrame a partir de um st.file_uploader (UploadedFile),
    detecta pela extensão e lê como Excel ou CSV com fallback de encoding.
    Salva em st.session_state.df e retorna o DataFrame.
    """
    # Pega nome e extensão do arquivo
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()
    df = None

    try:
        if suffix in {'.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'}:
            # Pandas aceita diretamente o UploadedFile para Excel
            df = pd.read_excel(uploaded_file)
        elif suffix == '.csv':
            # Para CSV, tenta utf-8 e depois latin1
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                # precisa reposicionar o cursor para o início
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            st.error(f"❌ Extensão não suportada: {suffix}")
            return None

    except FileNotFoundError:
        st.error(f"❌ Arquivo não encontrado: {filename}")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo: {e}")
        return None

    # Guarda no session_state e mostra sucesso
    st.session_state.df = df
    st.success(f"✅ Carregado: {filename} — {df.shape[0]} linhas × {df.shape[1]} colunas")
    return df

# === Uso no Streamlit ===


# Só roda a análise se o df já existir no session_state
secs = ["Análise descritiva", "Análise exploratória"]
tickers = secs
uploaded = st.sidebar.file_uploader(" ",
    type=["xlsx", "xls", "csv"])
ticker = st.sidebar.selectbox("Seções", tickers)

if uploaded is not None:
    present_file(uploaded)








