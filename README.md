# 💰 Predicción Salarial para Vacantes en Colombia

Esta es una aplicación web en **Streamlit** que permite a empleadores y reclutadores estimar el salario adecuado para una vacante laboral en Colombia. Utiliza un modelo de Machine Learning (`RandomForestRegressor`) entrenado con datos (simulados) del mercado laboral colombiano para predecir el mejor rango salarial en función a las características del empleo.

## 🗂️ Estructura del Proyecto

```text
.
├── app.py                # Aplicación principal y UI de Streamlit
├── train_model.py        # Script para generar datos y entrenar el modelo de ML
├── requirements.txt      # Archivo con los paquetes y versiones necesarias
├── README.md             # Documentación del proyecto
├── data/
│   └── sample_data.csv   # Dataset sintético (se genera al ejecutar train_model.py)
└── model/
    ├── preprocessor.pkl  # Objeto que transforma los inputs de texto/numéricos (se genera automáticamente)
    └── salary_model.pkl  # Modelo entrenado (se genera automáticamente)
```

## 🚀 Cómo ejecutar localmente

### 1. Clonar este repositorio
Si ya descargaste el código, dirígete a la carpeta raíz del proyecto desde tu terminal o consola de comandos.

### 2. Entorno virtual (Recomendado)
Es una buena práctica instalar las librerías en un entorno aislado de Python.
```bash
python -m venv venv
```
Actívalo dependiendo de tu sistema:
- **Windows:** `venv\Scripts\activate`
- **Linux/Mac:** `source venv/bin/activate`

### 3. Instalar dependencias
Usa el archivo suministrado para obtener las librerías exactas con `pip`.
```bash
pip install -r requirements.txt
```

### 4. Generar datos y entrenar el modelo
Es imprescindible tener el modelo en la carpeta `model/` para que la App lo consuma. 
Este comando generará 500 filas de ejemplos sintéticos para el mercado colombiano y entrenará el algoritmo.
```bash
python train_model.py
```
Verás que aparecen archivos nuevos en la carpeta `data/` y `model/`.

### 5. Iniciar la aplicación
Ejecuta la App con el siguiente comando (esto solucionará si el comando "streamlit" no es reconocido por problemas de PATH). Se abrirá automáticamente en tu navegador local (por defecto puerto `8501`).
```bash
python -m streamlit run app.py
```

## ☁️ Cómo desplegar en Streamlit Cloud

Desplegar esta app en la nube para que sea de uso público es muy sencillo a través de la plataforma de Streamlit.

1. **Sube tus archivos a un repositorio de GitHub.** 
   - Debes asegurarte de subir TODOS los archivos mencionados al inicio. En especial `app.py`, `requirements.txt` y los dos archivos de la carpeta `model/`. (No subas jamás la carpeta `venv/` ni cosas extrañas de tu SO, configúralo en un `.gitignore`).
2. **Ingresa a [Streamlit Community Cloud](https://share.streamlit.io/)**.
3. Inicia sesión usando la misma cuenta de GitHub donde de alojaron los archivos.
4. Presiona el botón verde **"New app"** (Nueva app).
5. Selecciona el repositorio, define la rama `main` (o `master`) y especifica **`app.py`** como el archivo general de ruta (*Main file path*).
6. Presiona **Deploy!**
7. En alrededor un minuto tu aplicación procesará todas las instalaciones y te entregará el link de tu app lista para usar en todo el mundo.
