import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

# Funciones auxiliares
def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

def calcular_metricas(df):
    returns = df.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    normalized_prices = df / df.iloc[0] * 100
    return returns, cumulative_returns, normalized_prices

def calcular_sesgo(df):
    return df.skew()

def calcular_exceso_curtosis(returns):
    return returns.kurtosis()

def calcular_rendimientos_portafolio(returns, weights):
    return (returns * weights).sum(axis=1)

def calcular_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calcular_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation != 0 else np.nan

def calcular_var_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    CVaR = returns[returns <= VaR].mean()
    return VaR, CVaR

def crear_histograma_distribucion(returns, var_95, cvar_95, title):
    fig = go.Figure()
    counts, bins = np.histogram(returns, bins=50)
    mask_before_var = bins[:-1] <= var_95

    fig.add_trace(go.Bar(
        x=bins[:-1][mask_before_var],
        y=counts[mask_before_var],
        width=np.diff(bins)[mask_before_var],
        name='Retornos < VaR',
        marker_color='rgba(255, 65, 54, 0.6)'
    ))

    fig.add_trace(go.Bar(
        x=bins[:-1][~mask_before_var],
        y=counts[~mask_before_var],
        width=np.diff(bins)[~mask_before_var],
        name='Retornos > VaR',
        marker_color='rgba(31, 119, 180, 0.6)'
    ))

    fig.add_trace(go.Scatter(
        x=[var_95, var_95],
        y=[0, max(counts)],
        mode='lines',
        name='VaR 95%',
        line=dict(color='green', width=2, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=[cvar_95, cvar_95],
        y=[0, max(counts)],
        mode='lines',
        name='CVaR 95%',
        line=dict(color='purple', width=2, dash='dot')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Retornos',
        yaxis_title='Frecuencia',
        showlegend=True,
        barmode='overlay',
        bargap=0
    )
    return fig

# Configuración de la página
st.set_page_config(page_title="Analizador de Portafolio", layout="wide")
st.sidebar.title("Analizador de Portafolio de Inversión")

# Configuración fija de ETFs permitidos y ventana de tiempo
etfs_permitidos = ["IEI", "EMB", "SPY", "IEMG", "GLD"]
start_date = "2010-01-01"
end_date = "2023-12-31"

# Entrada de símbolos y pesos
simbolos_input = st.sidebar.text_input(
    "Ingrese los símbolos de los ETFs (deben ser IEI, EMB, SPY, IEMG, GLD):", 
    ",".join(etfs_permitidos)
)
pesos_input = st.sidebar.text_input(
    "Ingrese los pesos correspondientes separados por comas (deben sumar 1):", 
    "0.2,0.2,0.2,0.2,0.2"
)

simbolos = [s.strip() for s in simbolos_input.split(',') if s.strip() in etfs_permitidos]
pesos = [float(w.strip()) for w in pesos_input.split(',')]

if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    st.sidebar.error("El número de símbolos debe coincidir con el número de pesos, y los pesos deben sumar 1.")
else:
    # Obtener datos
    df_stocks = obtener_datos_acciones(simbolos, start_date, end_date)
    returns, cumulative_returns, normalized_prices = calcular_metricas(df_stocks)
    
    # Rendimientos del portafolio
    portfolio_returns = calcular_rendimientos_portafolio(returns, pesos)
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Crear pestañas
    tab1, tab2 = st.tabs(["Análisis de Activos Individuales", "Análisis del Portafolio"])

    etf_summaries = {
    "IEI": {
        "nombre": "iShares 3-7 Year Treasury Bond ETF",
        "exposicion": "Bonos del Tesoro de EE. UU. con vencimientos entre 3 y 7 años",
        "indice": "ICE U.S. Treasury 3-7 Year Bond Index",
        "moneda": "USD",
        "pais": "Estados Unidos",
        "estilo": "Renta fija desarrollada",
        "costos": "0.15%",
    },
    "EMB": {
        "nombre": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
        "exposicion": "Bonos soberanos y cuasi-soberanos de mercados emergentes",
        "indice": "J.P. Morgan EMBI Global Core Index",
        "moneda": "USD",
        "pais": "Diversos mercados emergentes (Brasil, México, Sudáfrica, etc.)",
        "estilo": "Renta fija emergente",
        "costos": "0.39%",
    },
    "SPY": {
        "nombre": "SPDR S&P 500 ETF Trust",
        "exposicion": "500 empresas más grandes de Estados Unidos",
        "indice": "S&P 500 Index",
        "moneda": "USD",
        "pais": "Estados Unidos",
        "estilo": "Renta variable desarrollada",
        "costos": "0.09%",
    },
    "IEMG": {
        "nombre": "iShares Core MSCI Emerging Markets ETF",
        "exposicion": "Empresas de gran y mediana capitalización en mercados emergentes",
        "indice": "MSCI Emerging Markets Investable Market Index",
        "moneda": "USD",
        "pais": "China, India, Brasil, y otros mercados emergentes",
        "estilo": "Renta variable emergente",
        "costos": "0.11%",
    },
    "GLD": {
        "nombre": "SPDR Gold Shares",
        "exposicion": "Precio del oro físico (lingotes almacenados en bóvedas)",
        "indice": "Precio spot del oro",
        "moneda": "USD",
        "pais": "Exposición global",
        "estilo": "Materias primas",
        "costos": "0.40%",
    }
    }

    with tab1:
        st.header("Análisis de Activos Individuales")
        selected_asset = st.selectbox("Seleccione un activo para analizar:", simbolos)

          selected_asset = st.selectbox("Seleccione un activo para analizar:", simbolos)

        if selected_asset in etf_summaries:
            st.subheader(f"Resumen del ETF: {selected_asset}")
            summary = etf_summaries[selected_asset]
            st.markdown(f"""
            - **Nombre:** {summary['nombre']}
            - **Exposición:** {summary['exposicion']}
            - **Índice que sigue:** {summary['indice']}
            - **Moneda de denominación:** {summary['moneda']}
            - **País o región principal:** {summary['pais']}
            - **Estilo:** {summary['estilo']}
            - **Costos:** {summary['costos']}
            """)
        var_95, cvar_95 = calcular_var_cvar(returns[selected_asset])
        sharpe = calcular_sharpe_ratio(returns[selected_asset])
        sortino = calcular_sortino_ratio(returns[selected_asset])

        st.metric("Rendimiento Total", f"{cumulative_returns[selected_asset].iloc[-1]:.2%}")
        st.metric("VaR 95%", f"{var_95:.2%}")
        st.metric("CVaR 95%", f"{cvar_95:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Sortino Ratio", f"{sortino:.2f}")

        # Gráfico de precio normalizado
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[selected_asset], name=selected_asset))
        fig.update_layout(title=f"Precio Normalizado: {selected_asset}", xaxis_title="Fecha", yaxis_title="Precio Normalizado")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Análisis del Portafolio")
        portfolio_var_95, portfolio_cvar_95 = calcular_var_cvar(portfolio_returns)
        sharpe_portfolio = calcular_sharpe_ratio(portfolio_returns)
        sortino_portfolio = calcular_sortino_ratio(portfolio_returns)

        st.metric("Rendimiento Total del Portafolio", f"{portfolio_cumulative_returns.iloc[-1]:.2%}")
        st.metric("VaR 95% del Portafolio", f"{portfolio_var_95:.2%}")
        st.metric("CVaR 95% del Portafolio", f"{portfolio_cvar_95:.2%}")
        st.metric("Sharpe Ratio del Portafolio", f"{sharpe_portfolio:.2f}")
        st.metric("Sortino Ratio del Portafolio", f"{sortino_portfolio:.2f}")

        # Gráfico de rendimientos acumulados
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns, name="Portafolio"))
        fig.update_layout(title="Rendimientos Acumulados", xaxis_title="Fecha", yaxis_title="Rendimiento Acumulado")
        st.plotly_chart(fig, use_container_width=True)



        
