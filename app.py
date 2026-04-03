import streamlit as st
import pandas as pd
import joblib
import os

# ===== 1. CONFIGURACIÓN DE LA PÁGINA =====
st.set_page_config(
    page_title="Predicción Salarial Inteligente",
    page_icon="💼",
    layout="wide"
)

# ===== 2. LÓGICA Y FUNCIONES =====
@st.cache_resource
def load_models():
    """Carga de modelos almacenados con cache para máxima velocidad."""
    try:
        preprocessor = joblib.load('model/preprocessor.pkl')
        model = joblib.load('model/salary_model.pkl')
        return preprocessor, model
    except Exception as e:
        return None, None

def predict_salary(preprocessor, model, data_dict):
    """Ejecuta la inferencia sobre el input usando los modelos cargados."""
    input_df = pd.DataFrame([data_dict])
    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)[0]
    return prediction

preprocessor, model = load_models()

# Datos de referencia para las casillas
NIVELES = ['Junior', 'Semi-Senior', 'Senior', 'Líder']
CIUDADES = ['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Bucaramanga']
CONTRATOS = ['Indefinido', 'Fijo', 'Prestación de servicios']
MODALIDADES = ['Remoto', 'Híbrido', 'Presencial']
SECTORES = ['Tecnología', 'Finanzas', 'Retail', 'Salud', 'Educación', 'Manufactura']
EDUCACIONES = ['Técnico', 'Profesional', 'Especialización', 'Maestría']

# Data preconfigurada con fines de prueba
DEFAULT_DATA = {
    "cargo": "Científico de Datos Senior",
    "nivel": "Senior",
    "experiencia": 5,
    "ciudad": "Medellín",
    "tipo_contrato": "Indefinido",
    "modalidad": "Híbrido",
    "sector": "Tecnología",
    "educacion": "Profesional",
    "habilidades": "Python, SQL, AWS, Liderazgo"
}

# ===== 3. INTERFAZ: HEADER =====
st.write("") # Espaciador tope
col_logo, col_desc = st.columns([1, 4])

with col_logo:
    # Verificación de rutas de logo
    logo_paths = ["Media/logo.png", "media/logo.png", "Media/logo.jpg", "media/logo.jpg"]
    logo_loaded = False
    
    for path in logo_paths:
        if os.path.exists(path):
            st.image(path, width=220)
            logo_loaded = True
            break
            
    if not logo_loaded:
        st.info("Logo no encontrado en ruta local media/.")

with col_desc:
    st.title("Predicción Salarial Inteligente 📊")
    st.markdown("Optimiza tus procesos de selección con analítica de datos. Estima salarios de manera justa, competitiva y guiada por comportamientos históricos de Colombia.")

st.divider()

# ===== 4. INTERFAZ: DISEÑO A DOS COLUMNAS =====
col_input, col_espacio, col_output = st.columns([1.2, 0.1, 1])

with col_input:
    st.subheader("📋 Detalles de la Oferta")
    st.write("Configura los requisitos elementales correspondientes al candidato para el análisis.")
    
    # Manejo del estado del botón para pre-cargar datos
    if 'usar_ejemplo' not in st.session_state:
        st.session_state.usar_ejemplo = False

    if st.button("🔄 Cargar datos de ejemplo", use_container_width=True):
        st.session_state.usar_ejemplo = True

    is_ejemplo = st.session_state.usar_ejemplo
    
    # Formulario dentro de un Contenedor nativo emulando una "Tarjeta"
    with st.container(border=True):
        with st.form("salary_form", clear_on_submit=False):
            cargo = st.text_input("💻 Título del Puesto (Cargo)", value=DEFAULT_DATA["cargo"] if is_ejemplo else "")
            
            c1, c2 = st.columns(2)
            with c1:
                nivel = st.selectbox("Nivel Profesional", NIVELES, index=NIVELES.index(DEFAULT_DATA["nivel"]) if is_ejemplo else 0)
                experiencia = st.number_input("Años de Experiencia Mínima referidos", min_value=0, max_value=40, step=1, value=DEFAULT_DATA["experiencia"] if is_ejemplo else 0)
                ciudad = st.selectbox("Ciudad Operación", CIUDADES, index=CIUDADES.index(DEFAULT_DATA["ciudad"]) if is_ejemplo else 0)
                educacion = st.selectbox("Nivel Académico", EDUCACIONES, index=EDUCACIONES.index(DEFAULT_DATA["educacion"]) if is_ejemplo else 0)
            
            with c2:
                tipo_contrato = st.selectbox("Tipo de Vinculación", CONTRATOS, index=CONTRATOS.index(DEFAULT_DATA["tipo_contrato"]) if is_ejemplo else 0)
                modalidad = st.selectbox("Modalidad de Ejecución", MODALIDADES, index=MODALIDADES.index(DEFAULT_DATA["modalidad"]) if is_ejemplo else 0)
                sector = st.selectbox("Sector Económico", SECTORES, index=SECTORES.index(DEFAULT_DATA["sector"]) if is_ejemplo else 0)
                habilidades = st.text_input("Palabras clave extra / Skills (Opcional)", value=DEFAULT_DATA["habilidades"] if is_ejemplo else "")

            st.write("") # Espaciador ligero form interno
            submit_btn = st.form_submit_button("💰 Calcular salario corporativo", type="primary", use_container_width=True)

with col_output:
    st.subheader("📈 Resultado del Análisis Técnico")
    
    if submit_btn:
        # Validación
        if not cargo.strip():
            st.warning("⚠️ Para tener una estimación robusta requerimos que detalles el 'Título del Puesto'.")
        elif preprocessor is None or model is None:
            st.error("⚠️ Archivos analíticos desconectados o no hallados. Asegúrate de compilar `train_model.py` en tu ruta de trabajo antes de empezar.")
        else:
            with st.spinner("Calculando proyección métrica contra bases del mercado..."):
                input_dict = {
                    'cargo': cargo, 'nivel': nivel, 'experiencia': experiencia,
                    'ciudad': ciudad, 'tipo_contrato': tipo_contrato,
                    'modalidad': modalidad, 'sector': sector, 'educacion': educacion
                }

                try:
                    prediction = predict_salary(preprocessor, model, input_dict)
                    
                    lower_bound = prediction * 0.85
                    upper_bound = prediction * 1.15
                    
                    # Formato en miles COP nativo
                    formato_moneda = lambda x: f"${x:,.0f} COP".replace(",", ".")

                    st.success("Análisis algorítmico completado con éxito. Datos reflejados frente al marco coyuntural.")
                    
                    # 1. Métrica gigante principal nativa
                    with st.container(border=True):
                        st.metric(label="📌 Salario Mensual Sancionado", value=formato_moneda(prediction), delta="Base Fija Recomendada", delta_color="off")
                    
                    # 2. Visualización secundaria de los topes
                    with st.container(border=True):
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.write("**📉 Base piso aceptable:**")
                            st.subheader(formato_moneda(lower_bound))
                        with col_r:
                            st.write("**📈 Competitividad agresiva:**")
                            st.subheader(formato_moneda(upper_bound))
                    
                    # 3. Alerta interpretativa
                    st.info("""
                    **💡 Toma de decisiones de Talento:**
                    Este umbral de ±15% de control otorga un marco de seguridad operativa y negociación. Optar por ofertas en la escala alta del intervalo suele estar ligado a retención y captación estratégica frente a alta competencia de demanda en tu nicho.
                    """)
                    
                except Exception as e:
                    st.error(f"Falla crítica en procesamiento algorítmico: {str(e)}")
    else:
        # Estado Zero (Vacío)
        st.info("👈 Esperando variables: Utiliza el formulario izquierdo o carga los datos de demostración y aprieta el botón de cálculo.")
        st.write("")
        st.markdown("*Las métricas y dictámenes son actualizados bajo presiones estrictas de sector, grado tecnológico y localización geográfica para brindarte exactitud comparativa.*")
