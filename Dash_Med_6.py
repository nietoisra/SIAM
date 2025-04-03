## python -m streamlit run C:\Users\Nieto\Desktop\Dash_IA\Dash_Med_6.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, butter, filtfilt
import plotly.graph_objects as go
import os
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime


# Configuración de la página
st.set_page_config(
    page_title="SIAMIA - UAGro - Análisis de Marcha",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🚶‍♂️ Sistema Integral para el Análisis de Marcha con IA")
st.markdown("""
Sistema Integral para el Análisis de Marcha utilizando Inteligencia Artificial.
""")

# Divider
st.divider()

# Inicialización de variables de sesión
if 'paciente' not in st.session_state:
    st.session_state.paciente = None
if 'datos_paciente' not in st.session_state:
    st.session_state.datos_paciente = None
if 'metricas' not in st.session_state:
    st.session_state.metricas = None
if 'clasificacion' not in st.session_state:
    st.session_state.clasificacion = None
if 'anomalias' not in st.session_state:
    st.session_state.anomalias = None

# Funciones comunes
def butter_lowpass_filter(data, cutoff=3.0, fs=100, order=3):
    """Filtro pasa bajos para suavizar señales"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def detectar_pasos(señal, min_distance=35):
    """Detección de pasos en señal de aceleración"""
    peaks, _ = find_peaks(señal, height=np.mean(señal), distance=min_distance)
    return peaks

def calcular_metricas(tiempo, peaks, longitud_paso=0.7):
    """Cálculo de métricas clínicas de marcha"""
    if len(peaks) < 2:
        return None
    intervalos = np.diff(tiempo[peaks])
    cadencia = (1 / np.mean(intervalos)) * 60 * 2
    distancia = len(peaks) * longitud_paso
    velocidad = distancia / tiempo[-1] if tiempo[-1] > 0 else 0
    velocidad = velocidad * 2
    return {
        'pasos': len(peaks),
        'cadencia': cadencia,
        'distancia': distancia,
        'velocidad': velocidad,
        'asimetria': np.std(intervalos) / np.mean(intervalos) * 100,
        'intervalos': intervalos
    }

def interpretar_cadencia(valor, grupo):
    if grupo == "Adulto joven y saludable":
        if valor > 130:
            return ("Caminata rápida o actividad física intensa. Mantenga esta rutina si es intencional.", "blue")
        elif valor >= 95:
            return ("Normal. Su cadencia está dentro del rango saludable.", "green")
        else:
            return ("Por debajo del rango normal. Considere aumentar su actividad física o consulte a un especialista.", "orange")
    else:  # Adulto mayor
        if valor < 90:
            return ("Asociado con marcha lenta. Consulte a un especialista.", "red")
        elif valor <= 120:
            return ("Normal. Su cadencia está dentro del rango esperado para su edad.", "green")
        else:
            return ("Por encima del rango normal. Podría indicar una caminata rápida o actividad intensa.", "blue")

def interpretar_velocidad(valor, grupo):
    if grupo == "Adulto joven y saludable":
        if valor < 0.9:
            return ("Consulte a un especialista en movilidad.", "red")
        elif valor <= 1.5:
            return ("Normal. Su velocidad de marcha está dentro del rango saludable.", "green")
        else:
            return ("Caminata rápida o ejercicio. Mantenga esta rutina si es intencional.", "blue")
    else:  # Adulto mayor
        if valor < 0.9:
            return ("Realice ejercicios de fortalecimiento y equilibrio. Consulte a un especialista.", "orange")
        elif valor <= 1.2:
            return ("Normal. Su velocidad de marcha está dentro del rango esperado para su edad.", "green")
        else:
            return ("Caminata rápida o muy activo. Continúe con su rutina de actividad física.", "blue")

def interpretar_variabilidad(valor, grupo):
    if grupo == "Adulto joven y saludable":
        if valor > 0.10:
            return ("Consulte a un especialista para una evaluación más profunda.", "red")
        elif valor <= 0.06:
            return ("Normal. Su variabilidad de paso es consistente.", "green")
        else:
            return ("Ligeramente elevado. Considere realizar ejercicios de coordinación y equilibrio.", "orange")
    else:  # Adulto mayor
        if valor > 0.10:
            return ("Consulte a un especialista para una evaluación más profunda.", "red")
        else:  # Variabilidad <= 0.10
            return ("Normal. Su variabilidad de paso está dentro del rango esperado para su edad.", "green")

def generar_reporte_csv():
    """Genera archivo CSV con todos los resultados"""
    if not st.session_state.paciente or not st.session_state.metricas:
        return None
    # Calcular variabilidad de paso
    variabilidad = np.std(st.session_state.metricas['intervalos']) if 'intervalos' in st.session_state.metricas else 0
    reporte = {
        'Nombre': st.session_state.paciente['nombre'],
        'Edad': st.session_state.paciente['edad'],
        'Género': st.session_state.paciente['genero'],
        'Condición': st.session_state.paciente['condicion'],
        'Fecha_Registro': st.session_state.paciente['fecha_registro'],
        'Pasos_Detectados': st.session_state.metricas['pasos'],
        'Cadencia': st.session_state.metricas['cadencia'],
        'Velocidad': st.session_state.metricas['velocidad'],
        'Variabilidad_Pasos': variabilidad,
        'Asimetría': st.session_state.metricas['asimetria'],
        'Anomalías_Detectadas': st.session_state.anomalias['num_anomalias'] if st.session_state.anomalias else 0
    }
    if st.session_state.clasificacion:
        reporte['Clasificación'] = st.session_state.clasificacion['prediccion']
        reporte['Confianza'] = st.session_state.clasificacion['confianza']
        if 'reporte_clasificacion' in st.session_state.clasificacion:
            reporte.update(st.session_state.clasificacion['reporte_clasificacion'])
    df_reporte = pd.DataFrame([reporte])
    return df_reporte.to_csv(index=False).encode('utf-8')

# Detección de anomalías usando Isolation Forest
def detectar_anomalias(data):
    """Detecta anomalías en los datos de aceleración usando Isolation Forest"""
    modelo = IsolationForest(contamination=0.05, random_state=42)
    data_reshaped = data.reshape(-1, 1)  
    predicciones = modelo.fit_predict(data_reshaped)
    anomalias = data[predicciones == -1]  # Datos marcados como anomalías
    return anomalias, len(anomalias)

# Sección de Registro de Pacientes y Carga de Datos
st.write("""
## 📋 Registro de Personas y Carga de Datos
""")

# Inicializar st.session_state para la condición médica
if 'condicion_seleccionada' not in st.session_state:
    st.session_state.condicion_seleccionada = None

with st.form("paciente_form"):
    st.write("### Información del paciente")
    cols = st.columns(2)
    with cols[0]:
        nombre = st.text_input("Nombre completo*", help="Requerido para generar reportes")
        edad = st.number_input("Edad*", min_value=10, max_value=120, value=50)
    with cols[1]:
        genero = st.selectbox("Género*", ["Masculino", "Femenino"])
        # Campo para seleccionar la condición médica
        condicion = st.selectbox(
            "Condición médica", 
            ["Con Alzheimer", "Sin Alzheimer", "Otra"], 
            key="condicion_select"
        )
        # Almacenar la selección en st.session_state
        st.session_state.condicion_seleccionada = condicion

        # Mostrar campo de texto si se selecciona "Otra"
        if st.session_state.condicion_seleccionada == "Otra":
            condicion_otra = st.text_input("Especifique la condición médica*", help="Describa la condición médica")
        else:
            condicion_otra = None  # No se necesita descripción si no es "Otra"

    st.write("### Carga de datos de movimiento")
    uploaded_file = st.file_uploader("Suba el archivo CSV con datos de acelerómetro*", type=["csv"])
    
    if st.form_submit_button("Cargar datos y analizar"):
        # Validar campos obligatorios
        if not nombre or not uploaded_file:
            st.error("Por favor complete todos los campos obligatorios (*)")
        elif st.session_state.condicion_seleccionada == "Otra" and not condicion_otra:
            st.error("Por favor especifique la condición médica cuando seleccione 'Otra'")
        else:
            try:
                # Guardar datos del paciente
                condicion_final = condicion_otra if st.session_state.condicion_seleccionada == "Otra" else condicion
                st.session_state.paciente = {
                    'nombre': nombre,
                    'edad': edad,
                    'genero': genero,
                    'condicion': condicion_final,
                    'fecha_registro': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                # Cargar y validar datos del archivo
                data = pd.read_csv(uploaded_file)
                if not {'Tiempo', 'Acc X', 'Acc Y', 'Acc Z'}.issubset(data.columns):
                    st.error("El archivo debe contener las columnas: Tiempo, Acc X, Acc Y, Acc Z")
                else:
                    st.session_state.datos_paciente = data
                    st.success("Datos cargados correctamente. Procediendo con los análisis...")
            except Exception as e:
                st.error(f"Error al procesar el archivo: {str(e)}")

# Si hay datos cargados, proceder con los análisis
if st.session_state.datos_paciente is not None:
    data = st.session_state.datos_paciente
    # Análisis Clínico de Marcha
    st.divider()
    st.write("""
    ## 🏥 Análisis Clínico de Marcha
    """)
    # Parámetros fijos para el análisis
    longitud_paso = 0.7  # metros
    filtro_corte = 3.0    # Hz
    min_dist = 35         # muestras
    # Procesamiento
    tiempo = data['Tiempo'].values
    acc_y = data['Acc Y'].values
    fs = 1 / np.mean(np.diff(tiempo)) if len(tiempo) > 1 else 100
    señal_filtrada = butter_lowpass_filter(acc_y, cutoff=filtro_corte, fs=fs)
    peaks = detectar_pasos(señal_filtrada, min_distance=min_dist)
    # Cálculo de métricas
    metricas = calcular_metricas(tiempo, peaks, longitud_paso)
    if metricas:
        st.session_state.metricas = metricas
        # Mostrar resultados
        st.write("### Resultados del análisis de marcha")
        cols = st.columns(3)
        cols[0].metric("Pasos detectados", metricas['pasos'])
        cols[1].metric("Cadencia", f"{metricas['cadencia']:.1f} pasos/min")
        cols[2].metric("Velocidad", f"{metricas['velocidad']:.2f} m/s")
        # Gráfico de pasos detectados
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tiempo, 
            y=señal_filtrada, 
            name='Señal filtrada',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=tiempo[peaks], 
            y=señal_filtrada[peaks],
            mode='markers', 
            name='Pasos detectados',
            marker=dict(color='red', size=8)
        ))
        fig.update_layout(
            title="Detección de pasos en señal de aceleración",
            xaxis_title="Tiempo (segundos)",
            yaxis_title="Aceleración (m/s²)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detección de anomalías
    st.divider()
    st.write("""
    ## 🔍 Detección de Anomalías
    """)
    anomalias, num_anomalias = detectar_anomalias(acc_y)
    st.session_state.anomalias = {'anomalias': anomalias, 'num_anomalias': num_anomalias}
    st.write(f"### Número de anomalías detectadas: {num_anomalias}")
    #### Gráfico de anomalías
    fig_anomalias = go.Figure()
    fig_anomalias.add_trace(go.Scatter(
        x=tiempo, 
        y=acc_y, 
        name='Señal de aceleración',
        line=dict(color='blue')
    ))
    fig_anomalias.add_trace(go.Scatter(
        x=tiempo[np.isin(acc_y, anomalias)], 
        y=anomalias,
        mode='markers', 
        name='Anomalías detectadas',
        marker=dict(color='red', size=8)
    ))
    fig_anomalias.update_layout(
        title="Detección de anomalías en la señal de aceleración",
        xaxis_title="Tiempo (segundos)",
        yaxis_title="Aceleración (m/s²)",
        height=400
    )
    st.plotly_chart(fig_anomalias, use_container_width=True)

    # Clasificación de Marcha
    st.divider()
    st.write("""
    ## 🤖 Clasificación de Patrones de Marcha
    """)
    categorias = {'Lento': 'LENTO', 'Normal': 'NORMAL', 'Rápido': 'RAPIDO'}
    def cargar_datos_entrenamiento():
        datos = []
        for categoria, carpeta in categorias.items():
            ruta_carpeta = os.path.join('datos', carpeta)
            if not os.path.exists(ruta_carpeta):
                os.makedirs(ruta_carpeta)
                st.warning(f"Directorio '{ruta_carpeta}' creado. Por favor agregue datos de entrenamiento.")
                continue
            archivos_csv = [f for f in os.listdir(ruta_carpeta) if f.endswith('.csv')]
            for archivo in archivos_csv:
                ruta_archivo = os.path.join(ruta_carpeta, archivo)
                df_temp = pd.read_csv(ruta_archivo)
                df_temp['categoria'] = categoria
                datos.append(df_temp)
        return pd.concat(datos, ignore_index=True) if datos else None

    # Verificar modelo existente
    modelo_path = 'modelo_marcha_mejorado.pkl'
    columnas_path = 'columnas_modelo_mejorado.pkl'
    if os.path.exists(modelo_path) and os.path.exists(columnas_path):
        modelo = joblib.load(modelo_path)
        X_train_columns = joblib.load(columnas_path)
        st.success("Modelo de clasificación cargado exitosamente.")
    else:
        with st.spinner("Entrenando modelo automáticamente (esto puede tomar unos segundos)..."):
            # Parámetros fijos
            test_size = 0.10
            random_state = 42
            # Cargar datos de entrenamiento
            df_entrenamiento = cargar_datos_entrenamiento()
            if df_entrenamiento is None:
                st.error("No se encontraron datos de entrenamiento. Por favor agregue datos en las carpetas 'datos/LENTO', 'datos/NORMAL' y 'datos/RAPIDO'")
                st.stop()
            # Preparar datos
            X = df_entrenamiento.drop(columns=['categoria'])
            y = df_entrenamiento['categoria']
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                stratify=y, 
                random_state=random_state
            )
            # Balancear datos
            smote = SMOTE(random_state=random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            # Crear pipeline con escalado y modelo
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(random_state=random_state))
            ])
            # Parámetros para GridSearch
            param_grid = {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 10, 20],
                'clf__min_samples_split': [2, 5]
            }
            # Búsqueda de hiperparámetros
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train_balanced, y_train_balanced)
            # Mejor modelo
            modelo = grid_search.best_estimator_
            # Evaluación
            y_pred = modelo.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # Guardar modelo
            joblib.dump(modelo, modelo_path)
            joblib.dump(X_train.columns, columnas_path)
            st.success(f"Modelo entrenado automáticamente con precisión del {accuracy:.2%}")

    # Predicción con datos del paciente
    if modelo is not None and st.session_state.datos_paciente is not None:
        try:
            # Preparar datos para clasificación
            X_nuevos = st.session_state.datos_paciente.reindex(columns=X_train_columns, fill_value=0)
            # Realizar predicción
            prediccion = modelo.predict(X_nuevos)[0]
            probas = modelo.predict_proba(X_nuevos)[0]
            proba = np.max(probas) * 100
            # Guardar resultados
            st.session_state.clasificacion = {
                'prediccion': prediccion,
                'confianza': proba,
                'probabilidades': dict(zip(modelo.classes_, probas)),
                'reporte_clasificacion': {
                    f'Prob_{clase}': prob for clase, prob in zip(modelo.classes_, probas)
                }
            }
            # Mostrar resultados
            st.write("### Resultados de Clasificación")
            cols = st.columns(2)
            cols[0].metric("Patrón de marcha", prediccion)
            cols[1].metric("Confianza", f"{proba:.1f}%")
            # Gráfico de probabilidades
            st.write("### Distribución de Probabilidades")
            fig_proba = go.Figure()
            fig_proba.add_trace(go.Bar(
                x=modelo.classes_,
                y=probas * 100,
                marker_color=['red', 'green', 'blue'],
                text=[f"{p*100:.1f}%" for p in probas],
                textposition='auto'
            ))
            fig_proba.update_layout(
                yaxis_title="Probabilidad (%)",
                height=400
            )
            st.plotly_chart(fig_proba, use_container_width=True)
        except Exception as e:
            st.error(f"Error en la clasificación: {str(e)}")

    # Comparación con Valores Normales
    st.divider()
    st.write("""
    ## 📊 Comparación con Valores Normales
    """)
    if st.session_state.metricas and st.session_state.paciente:
        edad = st.session_state.paciente['edad']
        metricas = st.session_state.metricas
        intervalos = metricas['intervalos']
        variabilidad_paso = np.std(intervalos) if len(intervalos) > 1 else 0
        # Determinar grupo etario
        if edad < 60:
            grupo_etario = "Adulto joven y saludable"
            ref_cadencia = (100, 130)
            ref_velocidad = (1.1, 1.5)
            ref_variabilidad = 0.06  # Máximo normal
        else:
            grupo_etario = "Adulto mayor"
            ref_cadencia = (90, 120)
            ref_velocidad = (0.9, 1.2)
            ref_variabilidad = 0.10  # Máximo normal
        
        # Obtener interpretaciones
        cadencia_text, cadencia_color = interpretar_cadencia(metricas['cadencia'], grupo_etario)
        velocidad_text, velocidad_color = interpretar_velocidad(metricas['velocidad'], grupo_etario)
        variabilidad_text, variabilidad_color = interpretar_variabilidad(variabilidad_paso, grupo_etario)
        # Mostrar información del grupo etario
        st.write(f"#### Grupo: {grupo_etario} ({edad} años)")
        # Crear tabla de comparación
        df_comparacion = pd.DataFrame({
            'Parámetro': ['Cadencia (pasos/min)', 'Velocidad (m/s)', 'Variabilidad de paso (s)'],
            'Valor': [
                f"{metricas['cadencia']:.1f}",
                f"{metricas['velocidad']:.2f}",
                f"{variabilidad_paso:.3f}"
            ],
            'Rango Normal': [
                f"{ref_cadencia[0]}-{ref_cadencia[1]}",
                f"{ref_velocidad[0]}-{ref_velocidad[1]}",
                f"≤{ref_variabilidad:.2f}"
            ],
            'Interpretación': [cadencia_text, velocidad_text, variabilidad_text]
        })
        # Función para aplicar estilos
        def estilo_fila(row):
            colors = {
                'Parámetro': '',
                'Valor': '',
                'Rango Normal': '',
                'Interpretación': cadencia_color if row['Parámetro'] == 'Cadencia (pasos/min)' else 
                                velocidad_color if row['Parámetro'] == 'Velocidad (m/s)' else
                                variabilidad_color
            }
            return [f'color: {colors[col]}; font-weight: bold' if colors[col] else '' for col in row.index]
        # Aplicar estilos
        styled_df = df_comparacion.style.apply(estilo_fila, axis=1)
        # Mostrar tabla
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
        # Notas explicativas
        with st.expander("🔍 Explicación de los parámetros"):
            st.write("""
            - **Cadencia**: Número de pasos por minuto. Valores altos pueden indicar caminata rápida o actividad física intensa. 
            Valores bajos pueden sugerir deterioro motor.
            - **Velocidad**: Distancia recorrida por segundo (en metros por segundo). Valores bajos pueden estar asociados con 
            enfermedades neurológicas como Alzheimer.
            - **Variabilidad de paso**: Medida de fluctuación en la duración de pasos consecutivos (desviación estándar en segundos). 
            Valores altos (>0.10 s) se consideran patológicos y pueden indicar trastornos neurológicos.
            """)
        
        # Generar y ofrecer descarga del reporte
        st.divider()
        st.write("### 📥 Descargar Reporte CSV")
        csv_reporte = generar_reporte_csv()
        if csv_reporte:
            nombre_archivo = f"Reporte_{st.session_state.paciente['nombre'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
            st.download_button(
                label="Descargar Reporte en CSV",
                data=csv_reporte,
                file_name=nombre_archivo,
                mime="text/csv"
            )
        else:
            st.warning("No hay suficientes datos para generar el reporte")

from fpdf import FPDF
import base64
import tempfile

# Función para generar el PDF
def generar_reporte_pdf():
    """Genera un archivo PDF con todos los resultados y gráficas"""
    if not st.session_state.paciente or not st.session_state.metricas:
        return None

    # Crear un objeto PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Título del reporte
    pdf.cell(200, 10, txt="Reporte de Análisis de Marcha", ln=True, align="C")
    pdf.ln(10)

    # Información del paciente
    paciente = st.session_state.paciente
    pdf.cell(200, 10, txt=f"Nombre: {paciente['nombre']}", ln=True)
    pdf.cell(200, 10, txt=f"Edad: {paciente['edad']}", ln=True)
    pdf.cell(200, 10, txt=f"Género: {paciente['genero']}", ln=True)
    pdf.cell(200, 10, txt=f"Condición Médica: {paciente['condicion']}", ln=True)
    pdf.cell(200, 10, txt=f"Fecha de Registro: {paciente['fecha_registro']}", ln=True)
    pdf.ln(10)

    # Resultados de métricas
    metricas = st.session_state.metricas
    pdf.cell(200, 10, txt="Resultados del Análisis de Marcha:", ln=True)
    pdf.cell(200, 10, txt=f"Pasos Detectados: {metricas['pasos']}", ln=True)
    pdf.cell(200, 10, txt=f"Cadencia: {metricas['cadencia']:.1f} pasos/min", ln=True)
    pdf.cell(200, 10, txt=f"Velocidad: {metricas['velocidad']:.2f} m/s", ln=True)
    pdf.cell(200, 10, txt=f"Asimetría: {metricas['asimetria']:.2f}%", ln=True)
    pdf.ln(10)

    # Gráficas
    pdf.cell(200, 10, txt="Gráficas:", ln=True)
    pdf.ln(5)

    # Guardar gráficas como imágenes temporales
    def save_plot_as_image(fig, filename):
        fig.write_image(filename, format="png")

    # Gráfica de pasos detectados
    fig_pasos = go.Figure()
    tiempo = st.session_state.datos_paciente['Tiempo'].values
    señal_filtrada = butter_lowpass_filter(st.session_state.datos_paciente['Acc Y'].values)
    peaks = detectar_pasos(señal_filtrada)
    fig_pasos.add_trace(go.Scatter(x=tiempo, y=señal_filtrada, name='Señal filtrada', line=dict(color='blue')))
    fig_pasos.add_trace(go.Scatter(x=tiempo[peaks], y=señal_filtrada[peaks], mode='markers', name='Pasos detectados', marker=dict(color='red', size=8)))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        save_plot_as_image(fig_pasos, tmpfile.name)
        img_pasos = tmpfile.name

    # Gráfica de anomalías
    acc_y = st.session_state.datos_paciente['Acc Y'].values
    anomalias, _ = detectar_anomalias(acc_y)
    fig_anomalias = go.Figure()
    fig_anomalias.add_trace(go.Scatter(x=tiempo, y=acc_y, name='Señal de aceleración', line=dict(color='blue')))
    fig_anomalias.add_trace(go.Scatter(x=tiempo[np.isin(acc_y, anomalias)], y=anomalias, mode='markers', name='Anomalías detectadas', marker=dict(color='red', size=8)))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        save_plot_as_image(fig_anomalias, tmpfile.name)
        img_anomalias = tmpfile.name

    # Insertar imágenes en el PDF
    pdf.image(img_pasos, x=10, w=180)
    pdf.ln(10)
    pdf.image(img_anomalias, x=10, w=180)

    # Guardar el PDF
    pdf_output = pdf.output(dest="S").encode("latin1")
    return pdf_output

# Botón de descarga del PDF
if st.session_state.paciente and st.session_state.metricas:
    st.divider()
    st.write("### 📥 Descargar Reporte en PDF")
    pdf_reporte = generar_reporte_pdf()
    if pdf_reporte:
        nombre_archivo = f"Reporte_{st.session_state.paciente['nombre'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
        st.download_button(
            label="Descargar Reporte en PDF",
            data=pdf_reporte,
            file_name=nombre_archivo,
            mime="application/pdf"
        )
    else:
        st.warning("No hay suficientes datos para generar el reporte en PDF.")

# Footer
st.divider()
st.markdown("""
**Sistema Integral para el Análisis de Marcha utilizando Inteligencia Artificial** -- **SIAMIA** -- Versión 1.0  
**Herramienta para investigación biomecánica y análisis de patrones de marcha**  
© 2025 - UAGro - MIIDT
""")