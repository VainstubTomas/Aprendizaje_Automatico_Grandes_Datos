import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- 1. Cargar Modelos (Objetos .joblib) ---
try:
    # 1.1 Cargar el Scaler (RobustScaler ajustado a las 4 variables)
    scaler_kmeans = joblib.load('scaler_kmeans.joblib')
    # 1.2 Cargar el Modelo K-Means
    kmeans_model = joblib.load('kmeans_model.joblib')
except FileNotFoundError:
    st.error("Error: Archivos del modelo (.joblib) no encontrados. Asegúrate de guardarlos en la misma carpeta.")
    st.stop()

# 1.3 Mapeo de Clusters (Ajustado según tu orden de rendimiento: 0=BUENOS, 1=INTERMEDIOS, 2=MALOS)
# Debes confirmar el orden exacto de tus clusters, pero usaremos la siguiente lógica:
# El cluster con el performance_score más alto será "BUENOS".
# Asumo el orden más probable por la lógica de K-Means y tu análisis:
cluster_map = {
    # Estos índices (0, 1, 2) corresponden al índice del cluster en el df_model original.
    # El orden en tu Notebook era: cluster_order[0]: 'BUENOS', cluster_order[1]: 'INTERMEDIOS', cluster_order[2]: 'MALOS'
    2: 'BUENOS',        # El índice de cluster con mayor score (ejemplo)
    0: 'INTERMEDIOS',   # El índice de cluster intermedio (ejemplo)
    1: 'MALOS'          # El índice de cluster con menor score (ejemplo)
}
# Nota: La forma de verificar estos índices es revisando la salida de tu celda 36.

# --- 2. Configurar la Interfaz de Usuario ---
st.set_page_config(page_title="Predictor de Rendimiento", layout="centered")
st.title("VALIDA EL NIVEL DE RENDIMIENTO DE TU EQUIPO")
st.markdown("Los parametros de tu equipo serás comparados con la de equipos de alto rendimiento (La Liga Española)")
st.caption("Introduce el valor de tus parametros ofensivos, defensivos y tacticos de la temporada")

# 2.1 Inputs para los Scores
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        offensive = st.slider("Score Ofensivo:", 0.0, 1.0, 0.50, 0.01)
    with col2:
        defensive = st.slider("Score Defensivo:", 0.0, 1.0, 0.45, 0.01)
    with col3:
        tactical = st.slider("Score Táctico:", 0.0, 1.0, 0.40, 0.01)


# --- 3. Lógica de Predicción ---
if st.button("Definir Cluster de Rendimiento", type="primary"):
    
    # a) Calcular el Performance Score (Replicando tu fórmula de ponderación: 0.45, 0.40, 0.15)
    performance_score = (
        0.45 * offensive +
        0.40 * defensive +
        0.15 * tactical
    )
    
    # b) Preparar la entrada para el Scaler y el K-Means
    # ¡IMPORTANTE! El orden debe ser el mismo que en el entrenamiento: [Offensive, Defensive, Tactical, Performance]
    input_data_array = np.array([[offensive, defensive, tactical, performance_score]])
    
    # c) Escalar los datos de entrada
    X_input_scaled = scaler_kmeans.transform(input_data_array)
    
    # d) Predecir el Cluster
    predicted_cluster_index = kmeans_model.predict(X_input_scaled)[0]
    predicted_label = cluster_map.get(predicted_cluster_index, "Desconocido")
    
    # --- 4. Mostrar Resultados ---
    st.markdown("---")
    st.subheader("Resultados del Análisis:")
    
    st.metric(label="Performance Score (Global Calculado)", 
              value=f"{performance_score:.4f}", 
              delta_color="off")
    
    # 4.1 Mostrar el Cluster con estilo
    if predicted_label == 'BUENOS':
        st.success(f"🏆 El equipo se clasifica en el cluster: **{predicted_label}**")
    elif predicted_label == 'INTERMEDIOS':
        st.warning(f"🟡 El equipo se clasifica en el cluster: **{predicted_label}**")
    else:
        st.error(f"⬇️ El equipo se clasifica en el cluster: **{predicted_label}**")