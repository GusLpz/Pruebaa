import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

# Funciones auxiliares

def calcular_sesgo(df):
    return df.skew()

def calcular_exceso_curtosis(returns):
    return returns.kurtosis()

def calcular_ultimo_drawdown(series):
    peak = series.expanding(min_periods=1).max()
    drawdown = (series - peak) / peak
    ultimo_drawdown = drawdown.iloc[-1]
    return ultimo_drawdown
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
        sesgo = calcular_sesgo(returns[selected_asset])
        exceso_curtosis = calcular_exceso_curtosis(returns[selected_asset]) 
        ultimo_drawdown = calcular_ultimo_drawdown(cumulative_returns[selected_asset])
   

        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Total", f"{cumulative_returns[selected_asset].iloc[-1]:.2%}")
        col2.metric("Sharpe Ratio", f"{calcular_sharpe_ratio(returns[selected_asset]):.2f}")
        col3.metric("Sortino Ratio", f"{calcular_sortino_ratio(returns[selected_asset]):.2f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("VaR 95%", f"{var_95:.2%}")
        col5.metric("CVaR 95%", f"{cvar_95:.2%}")
        col6.metric("Media Retornos", f"{returns[selected_asset].mean():.2%}")
        
        col7, col8, col9 = st.columns(3)
        col7.metric("Sesgo de Retornos", f"{sesgo:.3f}")  # Nueva métrica
        col8.metric("Exceso de Curtosis", f"{exceso_curtosis:.3f}")  
        col9.metric("Último Drawdown", f"{ultimo_drawdown:.2%}")  # Último Drawdown añadido
        

    
        # Gráfico de precio normalizado
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[selected_asset], name=selected_asset))
        fig.update_layout(title=f"Precio Normalizado: {selected_asset}", xaxis_title="Fecha", yaxis_title="Precio Normalizado")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        portfolio_var_95, portfolio_cvar_95 = calcular_var_cvar(portfolio_returns)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Total del Portafolio", f"{portfolio_cumulative_returns.iloc[-1]:.2%}")
        col2.metric("Sharpe Ratio del Portafolio", f"{calcular_sharpe_ratio(portfolio_returns):.2f}")
        col3.metric("Sortino Ratio del Portafolio", f"{calcular_sortino_ratio(portfolio_returns):.2f}")

        col4, col5 = st.columns(2)
        col4.metric("VaR 95% del Portafolio", f"{portfolio_var_95:.2%}")
        col5.metric("CVaR 95% del Portafolio", f"{portfolio_cvar_95:.2%}")

        # Gráfico de rendimientos acumulados del portafolio vs benchmark
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns, name='Portafolio'))
        fig_cumulative.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[benchmark], name=selected_benchmark))
        fig_cumulative.update_layout(title=f'Rendimientos Acumulados: Portafolio vs {selected_benchmark}', xaxis_title='Fecha', yaxis_title='Rendimiento Acumulado')
        st.plotly_chart(fig_cumulative, use_container_width=True, key="cumulative_returns")


        # Beta del portafolio vs benchmark
        beta_portfolio = calcular_beta(portfolio_returns, returns[benchmark])
        st.metric(f"Beta del Portafolio vs {selected_benchmark}", f"{beta_portfolio:.2f}")

        st.subheader("Optimizacion de los portafolios")

        
        

        st.subheader("Distribución de Retornos del Portafolio vs Benchmark")
        
        col1, col2 = st.columns(2)
            
        with col1:
            # Histograma para el portafolio
            var_port, cvar_port = calcular_var_cvar(portfolio_returns)
            fig_hist_port = crear_histograma_distribucion(
                portfolio_returns,
                var_port,
                cvar_port,
                'Distribución de Retornos - Portafolio'
            )
            st.plotly_chart(fig_hist_port, use_container_width=True, key="hist_port")
            
        with col2:
            # Histograma para el benchmark
            var_bench, cvar_bench = calcular_var_cvar(returns[benchmark])
            fig_hist_bench = crear_histograma_distribucion(
                returns[benchmark],
                var_bench,
                cvar_bench,
                f'Distribución de Retornos - {selected_benchmark}'
            )
            st.plotly_chart(fig_hist_bench, use_container_width=True, key="hist_bench_2")

        # Rendimientos y métricas de riesgo en diferentes ventanas de tiempo
        st.subheader("Rendimientos y Métricas de Riesgo en Diferentes Ventanas de Tiempo")
        ventanas = [1, 7, 30, 90, 180, 252]
        
        # Crear DataFrames separados para cada métrica
        rendimientos_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])
        var_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])
        cvar_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [selected_benchmark])
        
        for ventana in ventanas:
            # Rendimientos
            rendimientos_ventanas[f'{ventana}d'] = pd.Series({
                'Portafolio': calcular_rendimiento_ventana(portfolio_returns, ventana),
                **{symbol: calcular_rendimiento_ventana(returns[symbol], ventana) for symbol in simbolos},
                selected_benchmark: calcular_rendimiento_ventana(returns[benchmark], ventana)
            })
            
            # VaR y CVaR
            var_temp = {}
            cvar_temp = {}
            
            # Para el portafolio
            port_var, port_cvar = calcular_var_cvar_ventana(portfolio_returns, ventana)
            var_temp['Portafolio'] = port_var
            cvar_temp['Portafolio'] = port_cvar
            
            # Para cada símbolo
            for symbol in simbolos:
                var, cvar = calcular_var_cvar_ventana(returns[symbol], ventana)
                var_temp[symbol] = var
                cvar_temp[symbol] = cvar
            
            # Para el benchmark
            bench_var, bench_cvar = calcular_var_cvar_ventana(returns[benchmark], ventana)
            var_temp[selected_benchmark] = bench_var
            cvar_temp[selected_benchmark] = bench_cvar
            
            var_ventanas[f'{ventana}d'] = pd.Series(var_temp)
            cvar_ventanas[f'{ventana}d'] = pd.Series(cvar_temp)
        
        # Mostrar las tablas
        st.subheader("Rendimientos por Ventana")
        st.dataframe(rendimientos_ventanas.style.format("{:.2%}"))
        
        st.subheader("VaR 95% por Ventana")
        st.dataframe(var_ventanas.style.format("{:.2%}"))
        
        st.subheader("CVaR 95% por Ventana")
        st.dataframe(cvar_ventanas.style.format("{:.2%}"))

        # Gráfico de comparación de rendimientos
        fig_comparison = go.Figure()
        for index, row in rendimientos_ventanas.iterrows():
            fig_comparison.add_trace(go.Bar(x=ventanas, y=row, name=index))
        fig_comparison.update_layout(title='Comparación de Rendimientos', xaxis_title='Días', yaxis_title='Rendimiento', barmode='group')
        # Gráfico de comparación de rendimientos
        st.plotly_chart(fig_comparison, use_container_width=True, key="returns_comparison")

        
