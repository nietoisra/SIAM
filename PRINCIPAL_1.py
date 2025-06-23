## python -m streamlit run C:\Users\Nieto\Desktop\TOTAL\PRINCIPAL_1.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import io
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
import joblib
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================== CONFIGURACIÓN INICIAL ==================
st.set_page_config(
    page_title="SIAM", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)
sns.set_style("whitegrid")
sns.set_palette("Set2")

# ================== FUNCIONES AUXILIARES COMPARTIDAS ==================

def butter_lowpass_filter(data, cutoff=5, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def filtrar_senales(df, columnas):
    for col in columnas:
        df[col] = butter_lowpass_filter(df[col])
    return df

def validate_data(df):
    """Validar que el archivo tenga las columnas requeridas"""
    required_columns = ['Tiempo', 'Acc X', 'Acc Y', 'Acc Z', 'Giro X', 'Giro Y']
    if not all(col in df.columns for col in required_columns):
        return False, f"El archivo debe contener las columnas: {', '.join(required_columns)}"
    
    # Verificar calidad de señal
    for col in required_columns[1:6]:
        if df[col].std() < 0.01:
            return False, f"La señal en {col} parece tener baja calidad o estar corrupta"
    
    return True, "Archivo válido"

def calculate_features(df):
    features = {}

    # Validar columnas necesarias
    required_cols = ['acc_x', 'acc_y', 'acc_z', 'giro_x', 'giro_y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Columna faltante en el DataFrame: {col}")

    # Media de aceleraciones
    features['acc_x_mean'] = df['acc_x'].mean()
    features['acc_y_mean'] = df['acc_y'].mean()
    features['acc_z_mean'] = df['acc_z'].mean()

    # Desviación estándar de aceleraciones
    features['acc_x_std'] = df['acc_x'].std()
    features['acc_y_std'] = df['acc_y'].std()
    features['acc_z_std'] = df['acc_z'].std()

    # Coeficiente de variación (%)
    features['acc_x_cv'] = (features['acc_x_std'] / abs(features['acc_x_mean'])) * 100 if features['acc_x_mean'] != 0 else float('inf')
    features['acc_y_cv'] = (features['acc_y_std'] / abs(features['acc_y_mean'])) * 100 if features['acc_y_mean'] != 0 else float('inf')
    features['acc_z_cv'] = (features['acc_z_std'] / abs(features['acc_z_mean'])) * 100 if features['acc_z_mean'] != 0 else float('inf')

    # CV promedio de los tres ejes
    features['acc_total_cv'] = np.mean([features['acc_x_cv'], features['acc_y_cv'], features['acc_z_cv']])

    # Media de giros
    features['giro_x_mean'] = df['giro_x'].mean()
    features['giro_y_mean'] = df['giro_y'].mean()

    # Desviación estándar de giros
    features['giro_x_std'] = df['giro_x'].std()
    features['giro_y_std'] = df['giro_y'].std()

    # Coeficiente de variación (%)
    features['giro_x_cv'] = (features['giro_x_std'] / abs(features['giro_x_mean'])) * 100 if features['giro_x_mean'] != 0 else float('inf')
    features['giro_y_cv'] = (features['giro_y_std'] / abs(features['giro_y_mean'])) * 100 if features['giro_y_mean'] != 0 else float('inf')

    # CV promedio de giros
    features['giro_total_cv'] = np.mean([features['giro_x_cv'], features['giro_y_cv']])

    # Índice de regularidad del movimiento (ejemplo simple basado en desvío entre pasos)
    if 'Tiempo' in df.columns:
        # Ejemplo: Calcular intervalos temporales entre cambios significativos (picos de aceleración)
        acc_magnitude = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        peaks = acc_magnitude[acc_magnitude > acc_magnitude.quantile(0.9)]
        times = df.loc[peaks.index, 'Tiempo']
        intervals = np.diff(times)
        if len(intervals) > 1:
            regularity = 1 / (1 + np.std(intervals))  # Mientras más bajo el std, más regular
            features['regularity_index'] = np.clip(regularity, 0, 1)
        else:
            features['regularity_index'] = 0.0
    else:
        features['regularity_index'] = 0.0  # Si no hay tiempo, se asume irregularidad máxima
    return features

# ================== ESTILOS PARA EL NUEVO MÓDULO ==================

EXPECTED_COLS_RAW = ['Tiempo', 'Acc X', 'Acc Y', 'Acc Z', 'Giro X', 'Giro Y']
STANDARD_COLS = ['Tiempo', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Giro_X', 'Giro_Y']

def validate_and_rename_columns(df):
    if all(col in df.columns for col in EXPECTED_COLS_RAW):
        df.columns = STANDARD_COLS
        return df
    elif list(df.columns) == list(range(len(STANDARD_COLS))):
        df.columns = STANDARD_COLS
        return df
    else:
        raise ValueError("Las columnas del archivo no coinciden con el formato esperado.")

def load_and_process_csv(file_content, filename):
    try:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        df = validate_and_rename_columns(df)
    except:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), header=None)
        df.columns = STANDARD_COLS

    df[STANDARD_COLS] = df[STANDARD_COLS].astype(float)
    df['Acc_Mag'] = np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)
    df['Giro_Mag'] = np.sqrt(df['Giro_X']**2 + df['Giro_Y']**2)
    return df

def calculate_features(data):
    features = {}
    acc_signal = data['Acc_Mag']
    features['acc_std'] = data['Acc_Mag'].std()
    features['acc_mean'] = data['Acc_Mag'].mean()
    features['acc_cv'] = features['acc_std'] / features['acc_mean'] if features['acc_mean'] != 0 else 0 
    features['giro_std'] = data['Giro_Mag'].std()
    features['giro_mean'] = data['Giro_Mag'].mean()
    features['giro_cv'] = features['giro_std'] / features['giro_mean'] if features['giro_mean'] != 0 else 0

    try:
        time_step = data['Tiempo'].diff().mean()
        freqs = fft.fftfreq(len(acc_signal), d=time_step)
        fft_vals = np.abs(fft.fft(acc_signal))
        dominant_freq = abs(freqs[np.argmax(fft_vals[1:len(fft_vals)//2]) + 1])
        features['step_frequency'] = dominant_freq
    except:
        features['step_frequency'] = 0

    try:
        acc_norm = (acc_signal - features['acc_mean']) / features['acc_std']
        autocorr = np.correlate(acc_norm, acc_norm, mode='full')[len(acc_norm)-1:]
        features['regularity'] = np.max(autocorr[1:50]) / autocorr[0] if autocorr[0] != 0 else 0
    except:
        features['regularity'] = 0
    return features

def create_comparison_plot(healthy_data, alzheimer_data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Promedio Sanos vs Alzheimer', fontsize=16, fontweight='bold')

    axes[0, 0].plot(healthy_data['Tiempo'], healthy_data['Acc_Mag'], label='Promedio Sanos', color='green', alpha=0.7)
    axes[0, 0].plot(alzheimer_data['Tiempo'], alzheimer_data['Acc_Mag'], label='Alzheimer', color='red', alpha=0.7)
    axes[0, 0].set_title('Aceleración vs Tiempo')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].hist(healthy_data['Acc_Mag'], bins=50, density=True, alpha=0.7, label='Promedio Sanos', color='green')
    axes[0, 1].hist(alzheimer_data['Acc_Mag'], bins=50, density=True, alpha=0.7, label='Alzheimer', color='red')
    axes[0, 1].set_title('Distribución de Aceleración')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    window_size = min(50, len(healthy_data)//10, len(alzheimer_data)//10)
    if window_size > 1:
        axes[1, 0].plot(healthy_data['Tiempo'], healthy_data['Acc_Mag'].rolling(window=window_size, min_periods=1).std(), label='Promedio Sanos', color='green')
        axes[1, 0].plot(alzheimer_data['Tiempo'], alzheimer_data['Acc_Mag'].rolling(window=window_size, min_periods=1).std(), label='Alzheimer', color='red')
        axes[1, 0].set_title('Variabilidad Móvil')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    h_feats = calculate_features(healthy_data)
    a_feats = calculate_features(alzheimer_data)
    metrics = ['Coef. Var.\nAcc', 'Coef. Var.\nGiro', 'Regularidad']
    keys = ['acc_cv', 'giro_cv', 'regularity']
    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 1].bar(x - width/2, [h_feats[k] for k in keys], width, label='Sano', color='green', alpha=0.7)
    axes[1, 1].bar(x + width/2, [a_feats[k] for k in keys], width, label='Alzheimer', color='red', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_title('Métricas Comparativas')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.close(fig)
    return fig

def modulo_comparativo_alzheimer():
    st.header("📊 Comparativo ALZ vs Sanos")

    if 'healthy_data' not in st.session_state:
        st.session_state.healthy_data = []
    if 'healthy_avg' not in st.session_state:
        st.session_state.healthy_avg = None
    if 'alzheimer_data' not in st.session_state:
        st.session_state.alzheimer_data = None

    st.subheader("1️⃣ Subir Archivos de Personas Sanas")
    healthy_files = st.file_uploader("Archivos CSV - Sujetos Sanos", type=['csv'], accept_multiple_files=True, key="healthy_files")

    if healthy_files:
        st.session_state.healthy_data = []
        for file in healthy_files:
            file_content = file.read()
            df = load_and_process_csv(file_content, file.name)
            st.session_state.healthy_data.append(df)

        all_features = [calculate_features(df) for df in st.session_state.healthy_data]
        st.session_state.healthy_avg = {
            key: np.mean([f[key] for f in all_features]) for key in all_features[0].keys()
        }
        st.success(f"Se procesaron {len(st.session_state.healthy_data)} archivos de personas sanas")

    st.subheader("2️⃣ Subir Archivo del Paciente con Alzheimer")
    alzheimer_file = st.file_uploader("Archivo CSV - Paciente ALZ", type=['csv'], key="alzheimer_file")

    if alzheimer_file:
        file_content = alzheimer_file.read()
        st.session_state.alzheimer_data = load_and_process_csv(file_content, alzheimer_file.name)
        st.success("Archivo Alzheimer procesado correctamente")

    if st.session_state.healthy_avg and st.session_state.alzheimer_data is not None:
        st.subheader("📈 Análisis Comparativo")

        alz_metrics = calculate_features(st.session_state.alzheimer_data)
        
        # Crear las métricas de comparación
        col1, col2, col3 = st.columns(3)
        
        # 1. Patrones de Movimiento (Aceleración)
        with col1:
            st.markdown("#### 🏃‍♂️ Patrones de Movimiento")
            acc_var_healthy = st.session_state.healthy_avg['acc_cv']
            acc_var_alz = alz_metrics['acc_cv']
            acc_diff_pct = ((acc_var_alz - acc_var_healthy) / acc_var_healthy) * 100
            
            st.metric(
                label="Variabilidad Aceleración (%)",
                value=f"{acc_var_alz:.2f}%",
                #delta=f"{acc_diff_pct:+.1f}% vs Sanos"
            )
            
            if acc_diff_pct > 30:
                st.error("⚠️ Movimientos muy irregulares")
            elif acc_diff_pct > 15:
                st.warning("⚠️ Movimientos algo irregulares")
            else:
                st.success("✅ Movimientos estables")

        # 2. Cambios de Dirección (Giro)
        with col2:
            st.markdown("#### 🔄 Cambios de Dirección")
            giro_var_healthy = st.session_state.healthy_avg['giro_cv']
            giro_var_alz = alz_metrics['giro_cv']
            giro_diff_pct = ((giro_var_alz - giro_var_healthy) / giro_var_healthy) * 100
            
            st.metric(
                label="Variabilidad Giro (%)",
                value=f"{giro_var_alz:.2f}%",
                #delta=f"{giro_diff_pct:+.1f}% vs Sanos"
            )
            
            if giro_diff_pct > 25:
                st.error("⚠️ Giros muy bruscos")
            elif giro_diff_pct > 10:
                st.warning("⚠️ Giros algo bruscos")
            else:
                st.success("✅ Giros controlados")

        # 3. Regularidad del Movimiento
        with col3:
            st.markdown("#### 📊 Regularidad")
            reg_healthy = st.session_state.healthy_avg['regularity']
            reg_alz = alz_metrics['regularity']
            reg_diff_pct = ((reg_alz - reg_healthy) / reg_healthy) * 100 if reg_healthy != 0 else 0
            
            st.metric(
                label="Índice de Regularidad",
                value=f"{reg_alz:.3f}",
                #delta=f"{reg_diff_pct:+.1f}% vs Sanos"
            )
            
            if abs(reg_diff_pct) > 10:
                st.warning("⚠️ Patrón irregular")
            else:
                st.success("✅ Patrón regular")

        # Resumen de Principales Diferencias
        st.subheader("🔍 Principales Diferencias Observadas")
        
        # Crear tarjetas de resumen
        differences_data = [
            {
                "Aspecto": "Patrones de Movimiento (Aceleración)",
                "Personas Sanas": "Movimientos más consistentes y estables",
                "Personas con Alzheimer": "Movimientos más irregulares y variables",
                "Diferencia Clave": f"{abs(acc_diff_pct):.1f}% {'más' if acc_diff_pct > 0 else 'menos'} variabilidad",
                "Diferencia_Num": abs(acc_diff_pct)
            },
            {
                "Aspecto": "Cambios de Dirección (Giro)",
                "Personas Sanas": "Giros más suaves y controlados",
                "Personas con Alzheimer": "Giros más bruscos e impredecibles",
                "Diferencia Clave": f"{abs(giro_diff_pct):.1f}% {'más' if giro_diff_pct > 0 else 'menos'} variabilidad",
                "Diferencia_Num": abs(giro_diff_pct)
            },
            {
                "Aspecto": "Regularidad del Movimiento",
                "Personas Sanas": "Patrones más regulares y predecibles",
                "Personas con Alzheimer": "Movimientos menos regulares",
                "Diferencia Clave": f"{abs(reg_diff_pct):.2f}% de diferencia",
                "Diferencia_Num": abs(reg_diff_pct)
            }
        ]
        
        # Mostrar las diferencias en formato expandible
        for i, diff in enumerate(differences_data, 1):
            with st.expander(f"{i}. {diff['Aspecto']}", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Personas Sanas:** {diff['Personas Sanas']}")
                with col_b:
                    st.markdown(f"**Personas con Alzheimer:** {diff['Personas con Alzheimer']}")
                
                # Color según la magnitud de la diferencia
                if diff['Diferencia_Num'] > 30:
                    st.error(f"🔴 **{diff['Diferencia Clave']}** - Diferencia muy significativa")
                elif diff['Diferencia_Num'] > 15:
                    st.warning(f"🟡 **{diff['Diferencia Clave']}** - Diferencia moderada")
                else:
                    st.info(f"🔵 **{diff['Diferencia Clave']}** - Diferencia leve")


        # Inicializar el estado del botón si no existe
        if "show_table" not in st.session_state:
            st.session_state.show_table = False

        # Botón para alternar visibilidad
        if st.button("🔄 Mostrar/Ocultar Tabla Comparativa"):
            st.session_state.show_table = not st.session_state.show_table

        # Mostrar la tabla si el estado es True
        if st.session_state.show_table:
            st.subheader("📋 Tabla Comparativa Detallada")
            comparison_data = []
            for key in st.session_state.healthy_avg:
                healthy_val = st.session_state.healthy_avg[key]
                alz_val = alz_metrics.get(key, 0)
                diff = alz_val - healthy_val
                diff_pct = (diff / healthy_val * 100) if healthy_val != 0 else 0

                comparison_data.append({
                    'Métrica': key.replace('_', ' ').title(),
                    'Promedio Sanos': f"{healthy_val:.4f}",
                    'Paciente ALZ': f"{alz_val:.4f}",
                    'Diferencia': f"{diff:+.4f}",
                    'Diferencia %': f"{diff_pct:+.2f}%"
                 })
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)


        # Visualización de señales comparativas
        st.subheader("📈 Comparación de Señales")
        if len(st.session_state.healthy_data) > 0:
            fig = create_comparison_plot(st.session_state.healthy_data[0], st.session_state.alzheimer_data)
            st.pyplot(fig)

        # Interpretación clínica
        st.subheader("🏥 Interpretación Comparativa")
        
        # Calcular puntuación de riesgo
        risk_score = 0
        risk_factors = []
        
        if acc_diff_pct > 30:
            risk_score += 3
            risk_factors.append("Variabilidad de movimiento muy alta")
        elif acc_diff_pct > 15:
            risk_score += 2
            risk_factors.append("Variabilidad de movimiento moderada")
        
        if giro_diff_pct > 25:
            risk_score += 3
            risk_factors.append("Giros muy bruscos e impredecibles")
        elif giro_diff_pct > 10:
            risk_score += 2
            risk_factors.append("Giros algo irregulares")
        
        if abs(reg_diff_pct) > 10:
            risk_score += 2
            risk_factors.append("Patrón de movimiento irregular")
        
        # Mostrar interpretación
        if risk_score >= 6:
            st.error("🔴 **ALTO RIESGO** - Patrón de marcha significativamente alterado")
            st.markdown("**Recomendaciones:**")
            st.markdown("- Evaluación neurológica inmediata")
            st.markdown("- Protocolo de prevención de caídas")
            st.markdown("- Supervisión constante durante la deambulación")
        elif risk_score >= 3:
            st.warning("🟡 **RIESGO MODERADO** - Alteraciones detectadas en el patrón de marcha")
            st.markdown("**Recomendaciones:**")
            st.markdown("- Seguimiento neurológico regular")
            st.markdown("- Fisioterapia especializada")
            st.markdown("- Monitoreo de la progresión")
        else:
            st.success("🟢 **RIESGO BAJO** - Patrón de marcha dentro de rangos esperados")
            st.markdown("**Recomendaciones:**")
            st.markdown("- Continuar con evaluaciones regulares")
            st.markdown("- Mantener actividad física")
        
        if risk_factors:
            st.markdown("**Factores de riesgo identificados:**")
            for factor in risk_factors:
                st.markdown(f"- {factor}")

    else:
        st.info("👆 Por favor, carga archivos de personas sanas y del paciente con Alzheimer para comenzar la comparación.")

# ================== MENÚ PRINCIPAL ==================

def main():
    st.title("🧠 SIAM")
    st.title("Sistema Integral de Análisis de Marcha")

    with st.sidebar:
        st.header("🔧 Selecciona el Módulo")
        modulo = st.selectbox("Elige un módulo para comenzar el análisis:", [
            "🏠 Inicio",
            "🤖 Clasificación ALZ vs SANO",
            "📊 Comparativo ALZ vs Sanos", 
            "🏥 Análisis de Marcha",
            "⚠️ Riesgo de Caída", 
        ])
        st.markdown("---")
        st.sidebar.header("📋 Archivos Esperados:")
        st.sidebar.markdown("""
            - Formato *.csv
            - Columnas: Tiempo, Acc X, Acc Y, Acc Z, Giro X, Giro Y
            """)
    
    if modulo == "🏠 Inicio":
        mostrar_inicio()
    elif modulo == "🤖 Clasificación ALZ vs SANO":
        modulo_prediccion()
    elif modulo == "📊 Comparativo ALZ vs Sanos":
        modulo_comparativo_alzheimer()
    elif modulo == "🏥 Análisis de Marcha":
        modulo_analisis_marcha()
    elif modulo == "⚠️ Riesgo de Caída":
        modulo_riesgo_caida()
    

def mostrar_inicio():
    #st.header("🏠 Bienvenido al Sistema Integral de Análisis de Marcha")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 Clasificación ALZ vs SANO
        - Clasificación automática usando ML
        - Análisis de múltiples archivos
        - Visualización de resultados
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Comparativo ALZ vs Sanos
        - Análisis de variabilidad
        - Comparación entre sujetos
        """)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        ### 🏥 Análisis de Marcha
        - Análisis de estabilidad de marcha
        - Factores de riesgo identificados
        - Recomendaciones clínicas
        """)
    
    with col4:
        st.markdown("""
        ### ⚠️ Riesgo de Caída
        - Análisis de variabilidad
        - Detección de movimientos súbitos
        - Comparación entre sujetos
        """)
# ================== MÓDULO 1: PREDICCIÓN ALZ vs SANO ==================

def butter_lowpass_filter(data, cutoff=5, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def filtrar_senales(df, columnas):
    for col in columnas:
        df[col] = butter_lowpass_filter(df[col])
    return df

def extraer_caracteristicas(df):
    caracteristicas = {}
    for eje in ["Acc X", "Acc Y", "Acc Z", "Giro X", "Giro Y"]:
        caracteristicas[f"{eje}_mean"] = np.mean(df[eje])
        caracteristicas[f"{eje}_std"] = np.std(df[eje])
        caracteristicas[f"{eje}_min"] = np.min(df[eje])
        caracteristicas[f"{eje}_max"] = np.max(df[eje])
        caracteristicas[f"{eje}_rms"] = np.sqrt(np.mean(df[eje]**2))
        hist, _ = np.histogram(df[eje], bins=10)
        caracteristicas[f"{eje}_entropy"] = entropy(hist + 1e-6)

    # Magnitud del vector de aceleración
    acc_mag = np.sqrt(df["Acc X"]**2 + df["Acc Y"]**2 + df["Acc Z"]**2)
    caracteristicas["acc_mag_mean"] = np.mean(acc_mag)
    caracteristicas["acc_mag_std"] = np.std(acc_mag)
    caracteristicas["acc_mag_max"] = np.max(acc_mag)

    # Cadencia aproximada
    peaks, _ = find_peaks(df["Acc Z"], height=np.mean(df["Acc Z"]) + 2 * df["Acc Z"].std())
    duracion_seg = df["Tiempo"].iloc[-1] - df["Tiempo"].iloc[0]
    caracteristicas["cadence"] = len(peaks) / (duracion_seg / 60)

    return caracteristicas

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Informe de Predicción de Marcha', align='C', ln=True)
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def add_dataframe(self, df):
        self.set_font("Arial", size=10)
        col_width = self.epw / len(df.columns)
        row_height = self.font_size * 1.5

        for i, row in df.iterrows():
            for datum in row:
                self.cell(col_width, row_height, str(datum), border=1)
            self.ln(row_height)

def generar_informe_pdf(nombre_archivo, prediccion, probabilidad, caracteristicas):
    pdf = PDF()
    pdf.add_page()

    # Información general
    pdf.chapter_title(f"Archivo: {nombre_archivo}")
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"clase_predicha: {prediccion}", ln=True)
    pdf.cell(0, 10, f"Probabilidad ALZ: {probabilidad[0]:.2%}", ln=True)
    pdf.cell(0, 10, f"Probabilidad SANO: {probabilidad[1]:.2%}", ln=True)
    pdf.ln(10)

    # Características extraídas
    pdf.chapter_title("Características Extraídas")
    df_features = pd.DataFrame([caracteristicas]).T.reset_index()
    df_features.columns = ["Característica", "Valor"]
    pdf.add_dataframe(df_features)

    try:
        p_alz = float(probabilidad[0])
        p_sano = float(probabilidad[1])
    except (IndexError, TypeError):
        p_alz = float(probabilidad)
        p_sano = 1.0 - p_alz

    pdf.cell(0, 10, f"Probabilidad ALZ: {p_alz:.2%}", ln=True)
    pdf.cell(0, 10, f"Probabilidad SANO: {p_sano:.2%}", ln=True)
    pdf.ln(10)

    df_features = pd.DataFrame([caracteristicas]).T.reset_index()
    df_features.columns = ["Característica", "Valor"]
    pdf.add_dataframe(df_features)

    return bytes(pdf.output(dest='S'))

@st.cache_resource
def cargar_modelo():
    modelo_path = modelo_clasificacion_marcha.pkl
    #modelo_path = "modelo_clasificacion_marcha.pkl"
    le_path = "label_encoder_marcha.pkl"
    try:
        modelo = joblib.load(modelo_path)
        le = joblib.load(le_path)
        return modelo, le
    except FileNotFoundError as e:
        st.error(f"No se encontró el archivo: {e}. Asegúrate de haber entrenado y guardado el modelo.")
        return None, None

def modulo_prediccion():
    st.header("🤖 Clasificación  ALZ vs SANO")
    
    modelo, le = cargar_modelo()
    if modelo is None or le is None:
        st.warning("⚠️ Modelo no encontrado. Este módulo requiere archivos de modelo entrenado.")
        st.info("Archivos necesarios: 'modelo_clasificacion_marcha.pkl' y 'label_encoder_marcha.pkl'")
        return

    uploaded_files = st.file_uploader("Sube uno o más archivos .csv", type="csv", accept_multiple_files=True)

    if uploaded_files:
        resultados = []
        total_archivos = len(uploaded_files)
        progress_bar = st.progress(0)

        for idx, file in enumerate(uploaded_files):
            df = pd.read_csv(file)
            file.seek(0)

            is_valid, mensaje = validate_data(df)
            if not is_valid:
                st.warning(f"Archivo {file.name}: {mensaje}")
                continue

            df_filtrado = filtrar_senales(df.copy(), ['Acc X', 'Acc Y', 'Acc Z', 'Giro X', 'Giro Y'])
            caracteristicas = extraer_caracteristicas(df_filtrado)
            X_nuevo = pd.DataFrame([caracteristicas])

            prediccion_codificada = modelo.predict(X_nuevo)[0]
            probabilidad = modelo.predict_proba(X_nuevo)[0]
            prediccion = le.inverse_transform([prediccion_codificada])[0]

            probabilidad_ALZ = f"{probabilidad[0] * 100:.2f}%"
            probabilidad_SANO = f"{probabilidad[1] * 100:.2f}%"

            resultados.append({
                "archivo": file.name,
                "clase_predicha": prediccion,
                "probabilidad_ALZ": probabilidad_ALZ,
                "probabilidad_SANO": probabilidad_SANO,
                "df_filtrado": df_filtrado,
                "caracteristicas": caracteristicas
            })
            progress_bar.progress(min((idx + 1) / total_archivos, 1.0))

        if resultados:
            df_resultados = pd.DataFrame(resultados)
            st.subheader("📊 Resultados de Clasificación")
            st.dataframe(df_resultados[['archivo', 'clase_predicha', 'probabilidad_ALZ', 'probabilidad_SANO']])

            st.subheader("📈 Distribución de Clases")
            fig = px.pie(df_resultados, names='clase_predicha', title='Distribución de Clases')
            st.plotly_chart(fig)
    else:
        st.info("👆 Sube uno o más archivos CSV para comenzar.")

# ================== MÓDULO 2: ANÁLISIS DE MARCHA ==================

def calculate_stats(values):
    return {
        'media': np.mean(values),
        'desviacion': np.std(values),
        'minimo': np.min(values),
        'maximo': np.max(values),
        'cv': (np.std(values) / abs(np.mean(values))) * 100 if np.mean(values) != 0 else 0
    }

def analyze_gait_stability(data):
    acc_x_stats = calculate_stats(data['Acc X'])
    acc_y_stats = calculate_stats(data['Acc Y'])
    acc_z_stats = calculate_stats(data['Acc Z'])
    giro_x_stats = calculate_stats(data['Giro X'])
    giro_y_stats = calculate_stats(data['Giro Y'])

    risk_factors = 0
    risk_details = []

    if acc_y_stats['desviacion'] > 1.0:
        risk_factors += 1
        risk_details.append("Inestabilidad lateral aumentada")
    if acc_z_stats['desviacion'] > 1.5:
        risk_factors += 1
        risk_details.append("Inestabilidad antero-posterior aumentada")
    if giro_x_stats['desviacion'] > 20 or giro_y_stats['desviacion'] > 20:
        risk_factors += 1
        risk_details.append("Movimientos compensatorios excesivos")

    return {
        'acc_x': acc_x_stats, 'acc_y': acc_y_stats, 'acc_z': acc_z_stats,
        'giro_x': giro_x_stats, 'giro_y': giro_y_stats,
        'risk_factors': risk_factors, 'risk_details': risk_details
    }

def create_risk_assessment(risk_factors):
    if risk_factors >= 2:
        return "RIESGO ALTO", "danger", "Requiere intervención inmediata"
    elif risk_factors == 1:
        return "RIESGO MODERADO", "warning", "Monitoreo recomendado"
    else:
        return "RIESGO BAJO", "success", "Patrón de marcha aceptable"

def text_to_pdf(report_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    for line in report_text.split('\n'):
        pdf.cell(0, 5, txt=line, ln=1)

    return bytes(pdf.output(dest='S'))

def modulo_analisis_marcha():
    st.header("🏥 Análisis de Marcha")
    
    with st.sidebar:
        st.subheader("Información del Paciente")
        nombre = st.text_input("Nombre del Paciente", value="Sujeto 1")
        edad = st.number_input("Edad", min_value=1, max_value=120, value=82)
        st.markdown("---")
    
    uploaded_file = st.file_uploader("Selecciona el archivo CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            is_valid, mensaje = validate_data(data)
            if not is_valid:
                st.error(mensaje)
                return
            data = data.dropna()
            duration = data['Tiempo'].max() - data['Tiempo'].min()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Paciente", f"{nombre}")
            with col2:
                st.metric("Edad", f"{edad} años")

            stability_analysis = analyze_gait_stability(data)
            tab1, tab2 = st.tabs(["🏥 Análisis de Marcha", "📋 Reporte"])
            
            with tab1:
                st.subheader("Evaluación de Marcha")
                risk_level, risk_color, risk_description = create_risk_assessment(stability_analysis['risk_factors'])

                if risk_color == "danger":
                    st.error(f"{risk_level} - {risk_description}")
                elif risk_color == "warning":
                    st.warning(f"{risk_level} - {risk_description}")
                else:
                    st.success(f"{risk_level} - {risk_description}")

                st.subheader(f"Factores de Riesgo Identificados: {stability_analysis['risk_factors']}/3")
                if stability_analysis['risk_details']:
                    for detail in stability_analysis['risk_details']:
                        st.write(f"• {detail}")
                else:
                    st.write("- Ninguno")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### Estabilidad Lateral")
                    lateral_std = stability_analysis['acc_y']['desviacion']
                    lateral_cv = stability_analysis['acc_y']['cv']
                    if lateral_std > 1.0:
                        st.error(f"⚠️ Alta variabilidad")
                    else:
                        st.success(f"✅ Estabilidad aceptable")
                    st.write(f"Desviación: {lateral_std:.3f} m/s²")
                    st.write(f"CV: {lateral_cv:.1f}%")

                with col2:
                    st.markdown("### Estabilidad Antero-Posterior")
                    ap_std = stability_analysis['acc_z']['desviacion']
                    ap_cv = stability_analysis['acc_z']['cv']
                    if ap_std > 1.5:
                        st.error(f"⚠️ Alta variabilidad")
                    else:
                        st.success(f"✅ Estabilidad aceptable")
                    st.write(f"Desviación: {ap_std:.3f} m/s²")
                    st.write(f"CV: {ap_cv:.1f}%")

                with col3:
                    st.markdown("### Movimientos Rotacionales")
                    giro_x_std = stability_analysis['giro_x']['desviacion']
                    giro_y_std = stability_analysis['giro_y']['desviacion']
                    if giro_x_std > 20 or giro_y_std > 20:
                        st.error(f"⚠️ Movimientos excesivos")
                    else:
                        st.success(f"✅ Rotación Controlada")
                    st.write(f"Giro X: {giro_x_std:.1f} °/s")
                    st.write(f"Giro Y: {giro_y_std:.1f} °/s")

                st.subheader("Recomendaciones Clínicas")
                recommendations = []

                if stability_analysis['risk_factors'] == 1:
                        recommendations = [
                        "Monitoreo regular del patrón de marcha",
                        "Evaluación del riesgo de caídas por especialista",
                        "Seguimiento neurológico continuo"
                ]
                elif stability_analysis['risk_factors'] == 2:
                        recommendations = [
                        "Fisioterapia enfocada en equilibrio",
                        "Supervisión durante la deambulación",
                        "Consideración de ayudas técnicas para la marcha"
                        ]
                elif stability_analysis['risk_factors'] == 3:
                        recommendations = [
                        "Evaluación inmediata por especialista",
                        "Protocolo de prevención de caídas",
                        "Revisión de medicación que pueda afectar el equilibrio"
                    ]

                for rec in recommendations:
                        st.write(f"• {rec}")

                with tab2:
                    st.subheader("Reporte Completo")
                    report = f"""

                    REPORTE DE ANÁLISIS DE MARCHA

                    INFORMACIÓN DEL PACIENTE
                    Nombre: {nombre}
                    Edad: {edad} años
                    Duración del Análisis: {duration:.2f} segundos
                    Número de Muestras: {len(data)}
                    
                    RESULTADOS DEL ANÁLISIS
                    
                    Estadísticas de Aceleración (m/s²):
                    Vertical (X): Media={stability_analysis['acc_x']['media']:.3f}, Std={stability_analysis['acc_x']['desviacion']:.3f}, CV={stability_analysis['acc_x']['cv']:.1f}%
                    Lateral (Y): Media={stability_analysis['acc_y']['media']:.3f}, Std={stability_analysis['acc_y']['desviacion']:.3f}, CV={stability_analysis['acc_y']['cv']:.1f}%
                    Antero-Posterior (Z): Media={stability_analysis['acc_z']['media']:.3f}, Std={stability_analysis['acc_z']['desviacion']:.3f}, CV={stability_analysis['acc_z']['cv']:.1f}%
                    
                    Estadísticas de Giro (°/s):
                    Giro X: Media={stability_analysis['giro_x']['media']:.3f}, Std={stability_analysis['giro_x']['desviacion']:.3f}
                    Giro Y: Media={stability_analysis['giro_y']['media']:.3f}, Std={stability_analysis['giro_y']['desviacion']:.3f}
                    
                    EVALUACIÓN DE RIESGO:
                    Nivel de Riesgo: {risk_level}
                    Factores de Riesgo: {stability_analysis['risk_factors']}/3
                    
                    Factores Identificados:
                    {chr(10).join([f"- {detail}" for detail in stability_analysis['risk_details']]) if stability_analysis['risk_details'] else "- Ninguno"}
                    
                    RECOMENDACIONES
                    {chr(10).join([f"- {rec}" for rec in recommendations])}
                    ---
                    Reporte generado automáticamente por el Sistema Integral de Análisis de Marcha
                    Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                                """
                    st.text_area("Contenido del Reporte", report, height=600)

                    try:
                        pdf_bytes = text_to_pdf(report)
                        st.download_button(
                            label="📄 Descargar Reporte (PDF)",
                            data=pdf_bytes,
                            file_name=f"reporte_marcha_{edad}años_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error al generar el PDF: {str(e)}")

        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
    else:
        st.info("👆 Por favor, carga un archivo CSV para comenzar")

# ================== MÓDULO 3: RIESGO DE CAÍDA ==================

def calculate_svm(df):
    return np.sqrt(df['Acc X']**2 + df['Acc Y']**2 + df['Acc Z']**2)

def calculate_std(arr):
    return np.std(arr)

def detect_peaks(signal, threshold):
    return [i for i in range(1, len(signal)-1) if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]]

def calculate_fall_risk_metrics(df):
    svm = calculate_svm(df)
    svm_mean, svm_std = svm.mean(), svm.std()

    acc_std_avg = df[['Acc X', 'Acc Y', 'Acc Z']].std().mean()
    gyro_std_avg = df[['Giro X', 'Giro Y']].std().mean()

    gvm = np.sqrt(df['Giro X']**2 + df['Giro Y']**2)
    gvm_mean, gvm_std = gvm.mean(), gvm.std()

    acc_peaks = detect_peaks(svm, svm_mean + 2 * svm_std)
    gyro_peaks = detect_peaks(gvm, gvm_mean + 2 * gvm_std)

    duration = df['Tiempo'].max() - df['Tiempo'].min()
    frequency = len(df) / duration if duration > 0 else 0

    return {
        'svm': {'mean': svm_mean, 'std': svm_std, 'data': svm},
        'accelerationVariability': acc_std_avg,
        'gyroscopeVariability': gyro_std_avg,
        'gyroscopeMagnitude': {'mean': gvm_mean, 'std': gvm_std, 'data': gvm},
        'suddenMovements': {
            'acceleration': len(acc_peaks),
            'rotation': len(gyro_peaks),
            'total': len(acc_peaks) + len(gyro_peaks),
            'acc_peaks': acc_peaks,
            'gyro_peaks': gyro_peaks
        },
        'dataQuality': {
            'duration': duration,
            'samples': len(df),
            'frequency': frequency
        }
    }

def assess_overall_risk(metrics):
    acc_risk = min(100, metrics['accelerationVariability'] * 20)
    giro_risk = min(100, metrics['gyroscopeVariability'] * 2)
    sudden_risk = min(100, metrics['suddenMovements']['total'] * 5)

    score = (acc_risk * 0.4 + giro_risk * 0.3 + sudden_risk * 0.3)
    level = 'BAJO' if score < 30 else 'MODERADO' if score < 60 else 'ALTO'

    return {
        'score': round(score),
        'level': level,
        'factors': {
            'Acc_Var': acc_risk,
            'Giro_Var': giro_risk,
            'Mov_Súbitos': sudden_risk
        }
    }


def modulo_riesgo_caida():
    st.header("⚠️ Riesgo de Caída")
    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            is_valid, mensaje = validate_data(df)
            if not is_valid:
                st.error(mensaje)
                return

            st.success("Archivo cargado exitosamente")

            metrics = calculate_fall_risk_metrics(df)
            risk = assess_overall_risk(metrics)

            st.subheader("Evaluación General")
            st.write(f"**Nivel de Riesgo:** {risk['level']}")

            # Gráfico de Factores de Riesgo Horizontal con Valores Visibles
            st.subheader("Factores de Riesgo")
            factors_df = pd.DataFrame(risk['factors'], index=['Riesgo (%)']).T.reset_index()
            factors_df.rename(columns={'index': 'Factor'}, inplace=True)

            fig_bar = px.bar(
                factors_df,
                x='Riesgo (%)',
                y='Factor',
                orientation='h',
                #title="Factores de Riesgo",
                text='Riesgo (%)',
                color='Riesgo (%)',
                color_continuous_scale='Bluered',
                range_color=[0, 100]
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300, template='plotly_white')
            fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig_bar)

            # Gráfico de Movimientos Súbitos usando Plotly (interactivo)
            st.subheader("Movimientos Súbitos en Aceleración")
            fig_line = go.Figure()

            fig_line.add_trace(go.Scatter(
                x=df['Tiempo'],
                y=metrics['svm']['data'],
                mode='lines',
                name='SVM (aceleración)',
                line=dict(color='#1f77b4')
            ))

            fig_line.add_trace(go.Scatter(
                x=df['Tiempo'].iloc[metrics['suddenMovements']['acc_peaks']],
                y=metrics['svm']['data'].iloc[metrics['suddenMovements']['acc_peaks']],
                mode='markers',
                name='Picos Detectados',
                marker=dict(color='red', size=8)
            ))

            fig_line.update_layout(
                #title='Detección de Movimientos Súbitos en Aceleración',
                xaxis_title='Tiempo (s)',
                yaxis_title='SVM',
                legend=dict(x=0.02, y=0.95),
                template='plotly_white'
            )
            st.plotly_chart(fig_line)

            # Análisis Comparativo
            st.subheader("Análisis Comparativo")
            multi_files = st.file_uploader("Sube archivos adicionales para comparar", type="csv", accept_multiple_files=True)

            if multi_files:
                summary = [{'Archivo': uploaded_file.name, 'Riesgo Global': risk['score'], 'Nivel': risk['level']}]

                for file in multi_files:
                    try:
                        df_tmp = pd.read_csv(file)
                        is_valid_tmp, _ = validate_data(df_tmp)
                        if is_valid_tmp:
                            metrics_tmp = calculate_fall_risk_metrics(df_tmp)
                            risk_tmp = assess_overall_risk(metrics_tmp)
                            summary.append({
                                'Archivo': file.name,
                                'Riesgo Global': round(risk_tmp['score'], 2),
                                'Nivel': risk_tmp['level']
                            })
                    except Exception as e:
                        st.warning(f"Error procesando {file.name}: {e}")

                summary_df = pd.DataFrame(summary)

                # Gráfico comparativo con colores por nivel
                color_map = {
                    'Bajo': '#2ecc71',
                    'Moderado': '#f1c40f',
                    'Alto': '#e74c3c'
                }

                fig_comp = px.bar(
                    summary_df,
                    x='Archivo',
                    y='Riesgo Global',
                    color='Nivel',
                    title='Comparación de Riesgo entre Sujetos',
                    color_discrete_map=color_map,
                    text='Riesgo Global'
                )
                fig_comp.update_traces(texttemplate='%{text}', textposition='outside')
                fig_comp.update_layout(xaxis_tickangle=-45, height=500, template='plotly_white')
                st.plotly_chart(fig_comp)

        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
    else:
        st.info("👆 Sube un archivo CSV para comenzar el análisis")

# ================== EJECUTAR APLICACIÓN ==================

if __name__ == "__main__":
    main()
