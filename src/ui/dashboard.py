import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import httpx
import time
from pathlib import Path
import os
import sys

# Configuración de la página
st.set_page_config(
    page_title="Risk Scoring Control Center",
    page_icon="🏦",
    layout="wide",
)

# Estilo CSS personalizado para Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4251;
    }
    .status-up { color: #00ff00; }
    .status-down { color: #ff0000; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURACIÓN Y CONSTANTES ---
API_URL = "http://localhost:8000"

# --- FUNCIONES DE UTILIDAD ---

def check_api_health():
    try:
        response = httpx.get(f"{API_URL}/health", timeout=2.0)
        return response.status_code == 200, response.json()
    except Exception:
        return False, None

def get_drift_status():
    try:
        response = httpx.get(f"{API_URL}/drift-status", timeout=2.0)
        return response.json()
    except Exception:
        return {"status": "Unknown", "message": "API not reachable"}

def call_predict(data):
    try:
        response = httpx.post(f"{API_URL}/predict", json=data, timeout=5.0)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

# --- SIDEBAR ---
st.sidebar.title("🏦 Menú de Navegación")
page = st.sidebar.radio("Ir a:", ["Dashboard de Monitoreo", "Simulador de Crédito", "Análisis de Drift"])

# Status de la API en el Sidebar
is_healthy, health_data = check_api_health()
if is_healthy:
    st.sidebar.success("● API ONLINE")
else:
    st.sidebar.error("○ API OFFLINE")

# --- CONTENIDO PRINCIPAL ---

if page == "Dashboard de Monitoreo":
    st.title("📊 Risk Scoring Control Center")
    st.markdown("---")
    
    # 1. Fila de KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicciones Totales", "5,432", "+12% vs ayer")
    
    with col2:
        st.metric("Tasa de Aprobación", "78.2%", "-2.1%")
        
    with col3:
        drift_info = get_drift_status()
        status_color = "🟢" if drift_info.get("status") == "Healthy" else "🔴"
        st.metric("Salud del Modelo", f"{status_color} {drift_info.get('status')}")
        
    with col4:
        st.metric("Latencia Media", "42ms", "-5ms")

    # 2. Gráficos de Distribución
    st.subheader("📈 Tendencias de Riesgo")
    c1, c2 = st.columns(2)
    
    with c1:
        # Simulación de datos para el gráfico
        df_dummy = pd.DataFrame({
            'Día': ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom'],
            'Aprobados': [120, 150, 140, 180, 200, 100, 90],
            'Rechazados': [20, 30, 25, 40, 50, 15, 10]
        })
        fig = px.bar(df_dummy, x='Día', y=['Aprobados', 'Rechazados'], 
                     title="Volumen de Solicitudes Semanales",
                     color_discrete_sequence=['#00cc96', '#ef553b'])
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        # Gráfico de Donut de Distribución
        labels = ['Bajo Riesgo', 'Medio Riesgo', 'Alto Riesgo']
        values = [4500, 600, 332]
        fig_donut = px.pie(names=labels, values=values, hole=.5,
                           title="Distribución de Perfiles de Riesgo",
                           color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_donut, use_container_width=True)

elif page == "Simulador de Crédito":
    st.title("🧪 Simulador de Crédito Real-Time")
    st.markdown("---")
    
    with st.form("prediction_form"):
        st.subheader("Datos del Solicitante")
        
        c1, c2 = st.columns(2)
        with c1:
            sk_id = st.number_input("ID Cliente", value=384575)
            income = st.number_input("Ingresos Totales (Anuales)", value=200000.0)
            credit = st.number_input("Monto del Crédito Solicitado", value=450000.0)
            annuity = st.number_input("Anualidad", value=50000.0)
            
        with c2:
            age_years = st.slider("Edad", 18, 90, 35)
            days_birth = age_years * -365 # El dataset usa días negativos
            
            emp_years = st.slider("Años en el empleo actual", 0, 50, 5)
            # El dataset usa días negativos para días empleados
            days_employed = emp_years * -365 
            
            education = st.selectbox("Nivel Educativo", 
                                   ["Secondary / secondary special", "Higher education", 
                                    "Incomplete higher", "Lower secondary", "Academic degree"])
            
            contract = st.selectbox("Tipo de Contrato", ["Cash loans", "Revolving loans"])

        submitted = st.form_submit_button("Analizar Riesgo 🚀")
        
        if submitted:
            if not is_healthy:
                st.error("❌ No se puede realizar la predicción: La API está offline.")
            else:
                input_data = {
                    "SK_ID_CURR": int(sk_id),
                    "AMT_INCOME_TOTAL": float(income),
                    "AMT_CREDIT": float(credit),
                    "AMT_ANNUITY": float(annuity),
                    "DAYS_BIRTH": int(days_birth),
                    "DAYS_EMPLOYED": int(days_employed),
                    "NAME_CONTRACT_TYPE": contract,
                    "NAME_EDUCATION_TYPE": education
                }
                
                with st.spinner("Consultando al Motor de Riesgo..."):
                    result = call_predict(input_data)
                
                if "error" in result:
                    st.error(f"Error en la API: {result['error']}")
                else:
                    st.markdown("---")
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.subheader("Resultado")
                        prob = result['probability']
                        risk_level = result['risk_level']
                        
                        if result['is_safe']:
                            st.success(f"### Score: {risk_level}")
                        else:
                            st.error(f"### Score: {risk_level}")
                            
                        st.write(f"**Probabilidad de Impago:** {prob:.2%}")
                    
                    with res_col2:
                        # Gráfico de Indicador (Gauge)
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Termómetro de Riesgo %"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "black"},
                                'steps' : [
                                    {'range': [0, 15], 'color': "lightgreen"},
                                    {'range': [15, 30], 'color': "yellow"},
                                    {'range': [30, 100], 'color': "red"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 15}
                            }
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)

elif page == "Análisis de Drift":
    st.title("🔬 Centro de Observabilidad y Drift")
    st.markdown("---")
    
    st.info("Aquí monitoreamos la consistencia estadística de nuestro flujo de datos frente al entrenamiento original.")
    
    drift_info = get_drift_status()
    st.write(f"**Estado del último reporte:** {drift_info.get('status')}")
    st.write(f"**Mensaje:** {drift_info.get('message')}")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📁 Reportes Generados")
        # Listar reportes en la carpeta reports/
        reports_dir = Path("reports")
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.html"))
            for rf in report_files:
                st.button(f"📄 Abrir {rf.name}", key=rf.name)
        else:
            st.write("No hay reportes disponibles.")
            
    with col_b:
        st.subheader("📊 Historial de Salud")
        # Simulación de histórico
        hist_df = pd.DataFrame({
            'Fecha': pd.date_range(start='2025-12-20', periods=7),
            'Drift Share %': [0.0, 0.01, 0.0, 0.02, 0.0, 0.15, 0.0]
        })
        fig_hist = px.line(hist_df, x='Fecha', y='Drift Share %', markers=True, 
                          title="Evolución del Drift Poblacional")
        st.plotly_chart(fig_hist, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("v1.0.0 | Powered by FastAPI, LightGBM & Streamlit")
