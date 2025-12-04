"""
CRYPTO NEXUS - Cyberpunk Big Data Analytics
Dashboard Web Interativo com Streamlit
Análise de Criptomoedas - Big Data
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion import CryptoDataIngestion
# from src.spark_processor import SparkDataProcessor
# from models.prophet_model import CryptoProphetModel, PROPHET_AVAILABLE
# from models.sklearn_model import CryptoMLModel, SKLEARN_AVAILABLE
from visualizations.plotly_charts import CryptoVisualizer, PLOTLY_AVAILABLE
from modern_theme import MODERN_CSS

# Configuração da página
st.set_page_config(
    page_title="📊 Crypto Analytics | Big Data Platform",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar tema moderno
st.markdown(MODERN_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data_from_yahoo(symbol, period="5y"):
    """Carrega dados do Yahoo Finance com cache"""
    ingestion = CryptoDataIngestion()
    df = ingestion.fetch_from_yahoo(symbol, period=period)
    return df


@st.cache_data
def process_data_spark(df):
    """Processa dados com Spark (com cache)"""
    processor = SparkDataProcessor()
    df_with_ma = processor.calculate_moving_averages(df)
    df_processed = processor.calculate_technical_indicators(df_with_ma)
    processor.stop()
    return df_processed


def main():
    # Header Moderno e Profissional
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0; margin-bottom: 1rem;'>
        <h1 style='font-size: 2.8em; font-weight: 700; margin: 0;'>
            💹 Crypto Analytics Platform
        </h1>
        <p style='font-family: "Inter", sans-serif; font-size: 1.1em; color: #A0A3AB; margin: 0.8rem 0 0 0; font-weight: 400;'>
            Big Data Processing • Machine Learning • Real-Time Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar Moderno
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(0, 102, 255, 0.3); margin-bottom: 1.5rem;'>
            <h2 style='font-family: "Poppins", sans-serif; color: #E8E9EB; font-weight: 600; margin: 0; font-size: 1.5em;'>
                ⚙️ Configuration
            </h2>
            <p style='font-family: "Inter", sans-serif; color: #A0A3AB; font-size: 0.9em; margin-top: 0.5rem; font-weight: 400;'>
                Data Source & Settings
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Seção: Fonte de Dados
        st.markdown("<p style='color: #0066FF; font-family: Inter; font-size: 0.9em; font-weight: 600; margin: 1.2rem 0 0.6rem 0;'>Data Source</p>", unsafe_allow_html=True)
        data_source = st.radio(
            "Data Source",
            ["🌐 Yahoo Finance API", "📁 Upload CSV File"],
            label_visibility="collapsed"
        )
        
        df = None
        
        if data_source == "🌐 Yahoo Finance API":
            st.markdown("<p style='color: #00B4D8; font-family: Inter; font-size: 0.85em; font-weight: 500; margin: 1rem 0 0.5rem 0;'>Cryptocurrency</p>", unsafe_allow_html=True)
            symbol = st.selectbox(
                "Crypto",
                ["₿ Bitcoin (BTC-USD)", "💎 Ethereum (ETH-USD)", "🪙 Cardano (ADA-USD)", "☀️ Solana (SOL-USD)", "🔶 Binance Coin (BNB-USD)", "💠 Ripple (XRP-USD)"],
                label_visibility="collapsed"
            )
            # Extrair símbolo
            symbol = symbol.split('(')[1].rstrip(')')
            
            st.markdown("<p style='color: #00B4D8; font-family: Inter; font-size: 0.85em; font-weight: 500; margin: 1rem 0 0.5rem 0;'>Time Period</p>", unsafe_allow_html=True)
            period = st.selectbox(
                "Period",
                ["📅 1 Year", "📅 2 Years", "📅 5 Years", "📅 10 Years", "📅 Maximum"],
                index=2,
                label_visibility="collapsed"
            )
            # Mapear para API
            period_map = {"📅 1 Year": "1y", "📅 2 Years": "2y", "📅 5 Years": "5y", "📅 10 Years": "10y", "📅 Maximum": "max"}
            period = period_map[period]
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Load Data", use_container_width=True, type="primary"):
                with st.spinner(f"Downloading data for {symbol}..."):
                    df = load_data_from_yahoo(symbol, period)
                    st.session_state['df'] = df
                    st.success(f"✅ Successfully loaded {len(df)} records")
        
        else:  # Upload CSV
            st.markdown("<p style='color: #06D6A0; font-family: Inter; font-size: 0.85em; font-weight: 500; margin: 1rem 0 0.5rem 0;'>Upload CSV File</p>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "CSV File",
                type=['csv'],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                try:
                    # Ler apenas uma amostra para arquivos muito grandes
                    st.info("📊 Carregando arquivo... Arquivos grandes serão amostrados.")
                    
                    # Ler primeira amostra para verificar tamanho
                    df_sample = pd.read_csv(uploaded_file, nrows=1000)
                    uploaded_file.seek(0)
                    
                    # Se arquivo é muito grande, usar amostragem
                    total_rows = sum(1 for _ in uploaded_file) - 1  # -1 para o header
                    uploaded_file.seek(0)
                    
                    if total_rows > 100000:
                        st.warning(f"⚠️ Arquivo tem {total_rows:,} linhas. Usando amostra de 100.000 linhas para melhor performance.")
                        df = pd.read_csv(uploaded_file, nrows=100000)
                    else:
                        df = pd.read_csv(uploaded_file)
                    
                    # Tentar detectar coluna de data
                    date_columns = ['Date', 'date', 'Datetime', 'datetime', 'timestamp', 'time']
                    date_col_found = None
                    
                    for col in date_columns:
                        if col in df.columns:
                            date_col_found = col
                            break
                    
                    if date_col_found:
                        df['Date'] = pd.to_datetime(df[date_col_found])
                        # Remover timezone se existir
                        if df['Date'].dt.tz is not None:
                            df['Date'] = df['Date'].dt.tz_localize(None)
                        if date_col_found != 'Date':
                            df = df.drop(columns=[date_col_found])
                    else:
                        # Se não encontrou, tentar usar o índice ou primeira coluna
                        st.warning("⚠️ Coluna de data não encontrada. Verificando primeira coluna...")
                        try:
                            # Tentar converter primeira coluna para data
                            first_col = df.columns[0]
                            df['Date'] = pd.to_datetime(df[first_col])
                            if df['Date'].dt.tz is not None:
                                df['Date'] = df['Date'].dt.tz_localize(None)
                            if first_col != 'Date':
                                df = df.drop(columns=[first_col])
                        except:
                            # Se falhar, usar índice como data
                            st.warning("⚠️ Criando índice temporal baseado na primeira linha...")
                            df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1min')
                    
                    # Verificar se tem coluna Close
                    if 'Close' not in df.columns:
                        price_cols = ['close', 'price', 'Price', 'value', 'Value']
                        for col in price_cols:
                            if col in df.columns:
                                df['Close'] = df[col]
                                break
                        
                        if 'Close' not in df.columns:
                            st.error("❌ Erro: Arquivo deve conter coluna 'Close' ou 'Price'")
                            st.stop()
                    
                    st.session_state['df'] = df
                    st.success(f"✅ {len(df)} registros carregados!")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
                    st.info("💡 Certifique-se que o CSV contém colunas: Date, Close (ou similares)")
        
        # Opções de processamento
        st.markdown("---")
        st.subheader("⚡ Processamento")
        
        if 'df' in st.session_state:
            if st.button("🔄 Processar com Spark"):
                with st.spinner("Processando dados..."):
                    df_processed = process_data_spark(st.session_state['df'])
                    st.session_state['df_processed'] = df_processed
                    st.success("✅ Dados processados!")
        
        # Informações do projeto
        st.markdown("---")
        st.subheader("ℹ️ Sobre")
        st.info("""
        **Projeto Big Data**
        
        Ferramentas utilizadas:
        - 🐍 Python 3.7+
        - ⚡ Apache Spark (PySpark)
        - 🔮 Prophet (Meta)
        - 🤖 scikit-learn
        - 📊 Plotly
        - 🌐 Streamlit
        
        Desenvolvido para análise de criptomoedas com Big Data.
        """)
    
    # Main content
    if 'df' not in st.session_state:
        st.info("👈 Selecione uma fonte de dados na barra lateral para começar.")
        
        # Instruções
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 Yahoo Finance")
            st.write("""
            - Escolha uma criptomoeda
            - Selecione o período
            - Clique em "Carregar Dados"
            """)
        
        with col2:
            st.markdown("### 📂 Upload CSV")
            st.write("""
            - Faça upload de um arquivo CSV
            - Deve conter colunas: Date, Close
            - Formatos aceitos: .csv
            """)
        
        return
    
    df = st.session_state['df']
    df_processed = st.session_state.get('df_processed', df)
    
    # Tabs principais
    tab1, tab2, tab4 = st.tabs([
        "📊 Overview",
        "📈 Technical Analysis",
        "📉 Visualizations"
    ])
    
    # Tab 1: Visão Geral
    with tab1:
        st.markdown('<p class="sub-header">📊 Visão Geral dos Dados</p>', unsafe_allow_html=True)
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        # Verificar se tem dados suficientes
        if len(df) < 2:
            st.warning("⚠️ Dados insuficientes para análise. Carregue mais registros.")
            return
        
        latest_price = df['Close'].iloc[-1]
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        max_price = df['Close'].max()
        min_price = df['Close'].min()
        
        with col1:
            st.metric(
                "💰 Preço Atual",
                f"${latest_price:,.2f}",
                f"{price_change:+.2f}%"
            )
        
        with col2:
            st.metric(
                "📈 Máximo",
                f"${max_price:,.2f}"
            )
        
        with col3:
            st.metric(
                "📉 Mínimo",
                f"${min_price:,.2f}"
            )
        
        with col4:
            volatility = df['Close'].pct_change().std() * 100
            st.metric(
                "📊 Volatilidade",
                f"{volatility:.2f}%"
            )
        
        # Informações do dataset
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Informações do Dataset")
            st.write(f"**Registros:** {len(df)}")
            st.write(f"**Colunas:** {', '.join(df.columns)}")
            st.write(f"**Período:** {df['Date'].min().date()} a {df['Date'].max().date()}")
        
        with col2:
            st.markdown("### 📊 Estatísticas Descritivas")
            st.dataframe(df[['Close', 'Volume']].describe() if 'Volume' in df.columns else df[['Close']].describe())
        
        # Preview dos dados
        st.markdown("---")
        st.markdown("### 👀 Preview dos Dados")
        st.dataframe(df.tail(20), use_container_width=True)
    
    # Tab 2: Análise Técnica
    with tab2:
        st.markdown('<p class="sub-header">📈 Análise Técnica</p>', unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            visualizer = CryptoVisualizer()
            
            # Gráfico de preços com MAs
            st.markdown("### 📊 Preço e Médias Móveis")
            fig_price = visualizer.plot_price_history(
                df_processed,
                show_ma=True,
                ma_windows=[7, 30, 90]
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Gráfico de volume
            if 'Volume' in df.columns:
                st.markdown("### 📊 Volume de Negociação")
                fig_volume = visualizer.plot_volume(df)
                st.plotly_chart(fig_volume, use_container_width=True)
            
            # Retornos
            if 'Daily_Return' in df_processed.columns:
                st.markdown("### 📉 Retornos Diários")
                fig_returns = visualizer.plot_returns(df_processed)
                st.plotly_chart(fig_returns, use_container_width=True)
            
            # Estatísticas das MAs
            st.markdown("---")
            st.markdown("### 📊 Médias Móveis Atuais")
            
            ma_cols = [col for col in df_processed.columns if col.startswith('MA_')]
            if ma_cols:
                ma_data = {}
                for col in ma_cols:
                    ma_data[col] = df_processed[col].iloc[-1]
                
                ma_df = pd.DataFrame(list(ma_data.items()), columns=['Média Móvel', 'Valor'])
                st.dataframe(ma_df, use_container_width=True)
        else:
            st.warning("⚠️ Plotly não está disponível. Instale com: pip install plotly")
    
    # Aba de Machine Learning foi desativada para a versão Light
    
    # Tab 4: Visualizações
    with tab4:
        st.markdown('<p class="sub-header">📉 Visualizações Avançadas</p>', unsafe_allow_html=True)
        
        if not PLOTLY_AVAILABLE:
            st.warning("⚠️ Plotly não está disponível")
            return
        
        visualizer = CryptoVisualizer()
        
        viz_type = st.selectbox(
            "Selecione o tipo de visualização:",
            [
                "Dashboard Completo",
                "Candlestick",
                "Matriz de Correlação"
            ]
        )
        
        if viz_type == "Dashboard Completo":
            fig_dashboard = visualizer.plot_dashboard(df_processed)
            st.plotly_chart(fig_dashboard, use_container_width=True)
        
        elif viz_type == "Candlestick":
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                # Filtro de período
                period_days = st.slider("Últimos N dias:", 30, 365, 90)
                df_recent = df.tail(period_days)
                
                fig_candlestick = visualizer.plot_candlestick(df_recent)
                st.plotly_chart(fig_candlestick, use_container_width=True)
            else:
                st.warning("⚠️ Dados OHLC não disponíveis")
        
        elif viz_type == "Matriz de Correlação":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            selected_cols = st.multiselect(
                "Selecione as colunas:",
                numeric_cols,
                default=list(numeric_cols[:5])
            )
            
            if selected_cols:
                fig_corr = visualizer.plot_correlation_heatmap(
                    df_processed,
                    columns=selected_cols
                )
                st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()
