

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math

# ----------------------------------------------------
# 1. PARÁMETROS y CARGA DE MODELOS y LÓGICA
# ----------------------------------------------------

# El orden de las 4 variables finales para el Scaler
COLUMNS_ORDER = ['offensive_score', 'defensive_score', 'tactical_score', 'performance_score']

# Mapeo de clusters (Asegúrate de que este mapeo es correcto, basado en el rendimiento: 
# BUENOS = mejor score, MALOS = peor score)
cluster_map = {
    0: 'BUENOS',        
    1: 'INTERMEDIOS',   
    2: 'MALOS'          
}

# Fórmulas Ponderadas (Tu lógica de cálculo de scores)
SCORE_FORMULAS = {
    # FACTORES OFENSIVOS
    'offensive_score': {
        'goals_pm': 0.40, 
        'xg_per_shot': 0.30, 
        'shots_pm': 0.15, 
        'goal_assists_pm': 0.10, 
        'shot_assists_pm': 0.05
    },
    # FACTORES DEFENSIVOS (Los negativos penalizan el score)
    'defensive_score': {
        'duel_win_rate': 0.40, 
        'counterpress_pm': 0.25, 
        'recovery_fail_pm': -0.20, # Penaliza si es alto
        'fouls_committed_pm': -0.10, # Penaliza si es alto
        'duel_foul_pm': -0.05 # Penaliza si es alto
    },
    # FACTORES TÁCTICOS
    'tactical_score': {
        'pattern_entropy': 0.30, 
        'tactics_pm': 0.20, 
        'pass_accuracy': 0.50
    }
}

# Ponderación del Score Global
GLOBAL_WEIGHTS = {
    'offensive_score': 0.45,
    'defensive_score': 0.40,
    'tactical_score': 0.15
}

# Carga de Modelos y Parámetros
try:
    scaler_kmeans = joblib.load('scaler_kmeans.joblib')
    kmeans_model = joblib.load('kmeans_model.joblib')
    minmax_params = joblib.load('minmax_params.joblib') 
except FileNotFoundError:
    st.error("Error: Archivos del modelo (.joblib) no encontrados. Asegúrate de guardar todos los archivos necesarios.")
    st.stop()


# Función CRÍTICA para replicar el cálculo de scores
def calculate_scores(input_metrics: dict, minmax_params: dict) -> dict:
    
    # 1. Función MinMax protegida
    def apply_minmax(value, min_val, max_val):
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    # 1. Normalización MinMax (0 a 1)
    normalized_metrics = {}
    for key, val in input_metrics.items():
        params = minmax_params.get(key)
        if params:
            normalized_metrics[key] = apply_minmax(val, params['min'], params['max'])
        else:
            normalized_metrics[key] = 0.0
            
    # 2. Cálculo de los 3 Sub-Scores (Sub-scores de 0 a 1)
    final_scores = {}
    for score_name, weights in SCORE_FORMULAS.items():
        score = 0
        for metric, weight in weights.items():
            # Suma ponderada de la métrica normalizada y su peso.
            # Los pesos (coeficientes) ya manejan la penalización (ej. -0.20)
            score += normalized_metrics.get(metric, 0.0) * weight
            
        # El resultado de esta suma es el score (ej. offensive_score)
        final_scores[score_name] = score

    # 3. Cálculo del Performance Score Global
    final_scores['performance_score'] = (
        final_scores['offensive_score'] * GLOBAL_WEIGHTS['offensive_score'] +
        final_scores['defensive_score'] * GLOBAL_WEIGHTS['defensive_score'] +
        final_scores['tactical_score'] * GLOBAL_WEIGHTS['tactical_score']
    )
    
    return final_scores

# ----------------------------------------------------
# 2. INTERFAZ DE USUARIO Y ENTRADA DE DATOS
# ----------------------------------------------------

st.set_page_config(page_title="Predictor de Rendimiento", layout="wide")
st.title("Clasificador de Rendimiento de Equipos")
st.markdown("Ingresa las metricas de tu equipo")
st.caption("Averigua que tipo de equipo es el tuyo comparado con la elite del futbol español")

# Diccionario para almacenar todos los inputs del usuario
user_inputs = {}

# --- A. Métricas Ofensivas ---
st.subheader("A. Ofensivas")
col1, col2, col3 = st.columns(3)
user_inputs['goals_pm'] = col1.number_input("Goles por Partido (goals_pm)", min_value=0.0, value=1.5, step=0.05, key='i_goals')
user_inputs['xg_per_shot'] = col2.number_input("xG por Tiro (xg_per_shot)", min_value=0.0, max_value=0.4, value=0.12, step=0.005, key='i_xgshot')
user_inputs['shots_pm'] = col3.number_input("Tiros por Partido (shots_pm)", min_value=0.0, value=12.0, step=0.5, key='i_shots')

colA, colB = st.columns(2)
user_inputs['goal_assists_pm'] = colA.number_input("Asistencias de Gol por Partido (goal_assists_pm)", min_value=0.0, value=1.0, step=0.05, key='i_gassist')
user_inputs['shot_assists_pm'] = colB.number_input("Asistencias de Tiro por Partido (shot_assists_pm)", min_value=0.0, value=8.0, step=0.5, key='i_sassist')


# --- B. Métricas Defensivas ---
st.subheader("B. Defensivas")
col1, col2, col3 = st.columns(3)
user_inputs['duel_win_rate'] = col1.number_input("Tasa de Duels Ganados", min_value=0.0, max_value=1.0, value=0.55, step=0.01, key='i_duelwin')
user_inputs['counterpress_pm'] = col2.number_input("Contrapresión por Partido", min_value=0.0, value=60.0, step=1.0, key='i_cpress')
user_inputs['recovery_fail_pm'] = col3.number_input("Recuperaciones Fallidas por Partido", min_value=0.0, value=3.0, step=0.1, key='i_recfail')

colA, colB = st.columns(2)
user_inputs['fouls_committed_pm'] = colA.number_input("Faltas Cometidas por Partido", min_value=0.0, value=12.0, step=0.5, key='i_foulcomm')
user_inputs['duel_foul_pm'] = colB.number_input("Faltas en Duels por Partido", min_value=0.0, value=1.5, step=0.1, key='i_duelfoul')

# --- C. Métricas Tácticas ---
st.subheader("C. Tácticas")
col1, col2, col3 = st.columns(3)
user_inputs['pattern_entropy'] = col1.number_input("Entropía de Patrón de Juego", min_value=0.0, max_value=1.0, value=0.65, step=0.01, key='i_entropy')
user_inputs['tactics_pm'] = col2.number_input("Ajustes Tácticos por Partido", min_value=0.0, value=2.0, step=0.1, key='i_tactics')
user_inputs['pass_accuracy'] = col3.number_input("Precisión de Pase", min_value=0.0, max_value=1.0, value=0.80, step=0.01, key='i_passacc')


# ----------------------------------------------------
# 4. LÓGICA DE CLASIFICACIÓN (al hacer click)
# ----------------------------------------------------

if st.button("Clasificar Equipo", type="primary"):
    
    # 4.1 Calcular Scores Intermedios y Global
    final_scores = calculate_scores(user_inputs, minmax_params)
    
    offensive_score = final_scores['offensive_score']
    defensive_score = final_scores['defensive_score']
    tactical_score = final_scores['tactical_score']
    performance_score = final_scores['performance_score']
    
    # 4.2 Preparar la entrada para el Scaler y el K-Means (4 scores)
    # Creamos el array con el orden de COLUMNS_ORDER
    input_array = np.array([offensive_score, defensive_score, tactical_score, performance_score]).reshape(1, -1)
    
    # 4.3 Escalar los datos de entrada
    X_input_scaled = scaler_kmeans.transform(input_array)
    
    # 4.4 Predecir el Cluster
    predicted_cluster_index = kmeans_model.predict(X_input_scaled)[0]
    predicted_label = cluster_map.get(predicted_cluster_index, "Cluster Desconocido")
    
    # ----------------------------------------------------
    # 5. MOSTRAR RESULTADOS
    # ----------------------------------------------------
    
    st.markdown("---")
    st.subheader("Scores Calculados y Clasificación Final")
    
    # Mostrar Scores Intermedios
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Score Ofensivo", f"{offensive_score:.4f}")
    colB.metric("Score Defensivo", f"{defensive_score:.4f}")
    colC.metric("Score Táctico", f"{tactical_score:.4f}")
    colD.metric("Performance Score Global", f"{performance_score:.4f}")
    
    st.markdown("---")
    
    # Mostrar el Cluster Final
    if predicted_label == 'BUENOS':
        st.success(f"🏆 Clasificación Final: **{predicted_label}**")
        st.balloons()
    elif predicted_label == 'INTERMEDIOS':
        st.warning(f"🟡 Clasificación Final: **{predicted_label}**")
    else:
        st.error(f"⬇️ Clasificación Final: **{predicted_label}**")