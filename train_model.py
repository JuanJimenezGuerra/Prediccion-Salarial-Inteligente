import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def generate_synthetic_data(num_samples=500):
    np.random.seed(42)
    
    niveles = ['Junior', 'Semi-Senior', 'Senior', 'Líder']
    ciudades = ['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Bucaramanga']
    contratos = ['Indefinido', 'Fijo', 'Prestación de servicios']
    modalidades = ['Remoto', 'Híbrido', 'Presencial']
    sectores = ['Tecnología', 'Finanzas', 'Retail', 'Salud', 'Educación', 'Manufactura']
    educaciones = ['Técnico', 'Profesional', 'Especialización', 'Maestría']
    cargos = ['Arquitecto Cloud', 'Científico de Datos', 'Desarrollador Backend', 'Desarrollador Frontend', 
              'Analista QA', 'DevOps', 'Gerente de TI', 'Diseñador UI/UX']
    
    data = []
    for _ in range(num_samples):
        cargo = np.random.choice(cargos)
        nivel = np.random.choice(niveles)
        
        if nivel == 'Junior':
            experiencia = np.random.randint(0, 3)
        elif nivel == 'Semi-Senior':
            experiencia = np.random.randint(2, 6)
        elif nivel == 'Senior':
            experiencia = np.random.randint(5, 10)
        else: # Líder
            experiencia = np.random.randint(8, 15)
            
        ciudad = np.random.choice(ciudades)
        tipo_contrato = np.random.choice(contratos)
        modalidad = np.random.choice(modalidades)
        sector = np.random.choice(sectores)
        educacion = np.random.choice(educaciones)
        
        salario_base = 1800000 
        
        mult_nivel = {'Junior': 1.0, 'Semi-Senior': 1.8, 'Senior': 3.2, 'Líder': 4.8}[nivel]
        mult_exp = 1.0 + (experiencia * 0.04)
        mult_ciudad = {'Bogotá': 1.25, 'Medellín': 1.15, 'Cali': 1.05, 'Barranquilla': 1.0, 'Bucaramanga': 0.9}[ciudad]
        mult_edu = {'Técnico': 1.0, 'Profesional': 1.2, 'Especialización': 1.5, 'Maestría': 1.8}[educacion]
        mult_modalidad = {'Remoto': 1.1, 'Híbrido': 1.05, 'Presencial': 1.0}[modalidad]
        mult_sector = {'Tecnología': 1.3, 'Finanzas': 1.25, 'Salud': 1.1, 'Manufactura': 1.05, 'Educación': 0.9, 'Retail': 0.95}[sector]
        
        salario = salario_base * mult_nivel * mult_exp * mult_ciudad * mult_edu * mult_modalidad * mult_sector
        ruido = np.random.normal(0, 0.12)
        salario = salario * (1 + ruido)
        
        salario = round(salario, -4) 
        
        data.append([cargo, nivel, experiencia, ciudad, tipo_contrato, modalidad, sector, educacion, salario])
        
    df = pd.DataFrame(data, columns=['cargo', 'nivel', 'experiencia', 'ciudad', 'tipo_contrato', 'modalidad', 'sector', 'educacion', 'salario'])
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_data.csv', index=False)
    print("✅ Datos sintéticos generados en 'data/sample_data.csv'")
    return df

def train():
    print("Iniciando proceso de entrenamiento...")
    df = generate_synthetic_data()
    
    X = df.drop('salario', axis=1)
    y = df['salario']
    
    cat_cols = ['cargo', 'nivel', 'ciudad', 'tipo_contrato', 'modalidad', 'sector', 'educacion']
    num_cols = ['experiencia']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"✅ Modelo entrenado exitosamente. R^2 score: {score:.4f}")
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(preprocessor, 'model/preprocessor.pkl')
    joblib.dump(model, 'model/salary_model.pkl')
    print("✅ Modelos y transformadores guardados en 'model/'")

if __name__ == '__main__':
    train()
