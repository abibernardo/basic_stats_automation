
from pathlib import Path
import streamlit as st
import pandas as pd

import streamlit as st

st.set_page_config(
    page_title="Análise de Dados",
    page_icon="📊",
    layout="wide"
)

# --- Estilo CSS personalizado ---
st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-top: 40px;
}

.subtitle {
    font-size: 22px;
    font-weight: 400;
    text-align: center;
    color: #555;
    margin-top: -10px;
}

.author-box {
    text-align: center;
    font-size: 18px;
    margin-top: 25px;
    color: #444;
}

.divider {
    height: 2px;
    margin: 35px auto 25px auto;
    background: linear-gradient(
        90deg,
        rgba(0,0,0,0) 0%,
        rgba(90,90,90,0.4) 50%,
        rgba(0,0,0,0) 100%
    );
    width: 70%;
    border-radius: 2px;
}

.description {
    text-align: center;
    font-size: 19px;
    max-width: 850px;
    margin: 0 auto;
    line-height: 1.6;
    color: #333;
}

</style>
""", unsafe_allow_html=True)


# --- Conteúdo visual ---
st.markdown(
    "<div class='main-title'>Aplicação Interativa para Análise de Dados</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
Tratamento, exploração e análise do seu conjunto de dados!
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='author-box'>
<strong>Desenvolvido por:</strong> Bernardo Almeida  
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div class='description'>
Carregue arquivos <strong>Excel</strong> ou <strong>CSV</strong>,
e faça a análise do seus dados de forma simples e interativa - sem precisar escrever uma única linha de código!
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)


def present_file(uploaded_file):
    """
    Carrega um DataFrame a partir de um st.file_uploader (UploadedFile),
    detecta pela extensão e lê como Excel ou CSV com fallback de encoding.
    Salva em st.session_state.df e retorna o DataFrame.
    """
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()
    df = None

    try:
        if suffix in {'.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'}:
            df = pd.read_excel(uploaded_file)

        elif suffix == '.csv':
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
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

    st.session_state.df = df
    st.success(
        f"✅ **Arquivo carregado com sucesso!**  \n"
        f"📄 `{filename}`  \n"
        f"📐 {df.shape[0]} linhas × {df.shape[1]} colunas"
    )
    return df


# -----------------------------
# Área de upload
# -----------------------------
with st.container():
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        uploaded = st.file_uploader(
            label="📎 Selecione o arquivo",
            type=["xlsx", "xls", "csv"]
        )

if uploaded is not None:
    present_file(uploaded)

