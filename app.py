from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, roc_curve, auc,
                            mean_absolute_error)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Configuración CORS más específica

# Variables globales para el modelo
model = None
scaler = None
selector = None
features = None
metrics = {}

def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Semestre académico que cursa actualmente": "Semestre",
        "¿Cuántas horas estudias al día en promedio (fuera de clase)?": "Horas_estudio",
        "¿Cuántas horas duermes por noche en promedio?": "Horas_sueno",
        "¿Cuántas horas al día usas redes sociales (WhatsApp, Instagram, TikTok, etc.)?": "Horas_redes",
        "¿Consumes bebidas con cafeína regularmente (café, energizantes, té)?": "Cafeina",
        "¿Cuál es tu promedio académico actual (sobre 5.0)?": "Promedio",
        "En una escala del 1 al 10, ¿qué tanto estrés sientes en tu vida académica actualmente?": "Nivel_estres"
    })

    # Limpieza de outliers
    df = df[(df['Horas_sueno'] > 2) & (df['Horas_sueno'] < 12)]
    df = df[(df['Horas_estudio'] >= 0) & (df['Horas_estudio'] < 15)]
    df = df[(df['Promedio'] >= 1) & (df['Promedio'] <= 5)]
    df = df[(df['Horas_redes'] >= 0) & (df['Horas_redes'] < 15)]

    return df

def clasificar_estres(nivel):
    if nivel <= 3:
        return 0  # Bajo
    elif nivel <= 6:
        return 1  # Medio
    else:
        return 2  # Alto

def preprocesar_datos(df):
    df['Estres_cat'] = df['Nivel_estres'].apply(clasificar_estres)
    df['Cafeina'] = df['Cafeina'].astype(str).str.strip().str.lower().map({
        'sí': 1, 'si': 1, 'yes': 1, '1': 1, 'no': 0, '0': 0}).fillna(0).astype(int)

    features = ["Semestre", "Horas_estudio", "Horas_sueno", "Horas_redes",
                "Cafeina", "Promedio"]

    imputer = SimpleImputer(strategy='median')
    df[features] = imputer.fit_transform(df[features])

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    y = df["Estres_cat"]

    selector = SelectKBest(f_classif, k=6)
    X_selected = selector.fit_transform(X, y)

    return X_selected, y, scaler, selector, features

def entrenar_modelo(X, y):
    smt = SMOTETomek(random_state=42)
    X_bal, y_bal = smt.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, stratify=y_bal, random_state=42)

    model = XGBClassifier(
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=300,
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model, X_test, y_test

def get_model_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Bajo", "Medio", "Alto"], output_dict=True)
    
    # Generar matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Generar curvas ROC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'roc_curves': {
            'fpr': {str(i): fpr[i].tolist() for i in range(3)},
            'tpr': {str(i): tpr[i].tolist() for i in range(3)},
            'auc': {str(i): roc_auc[i] for i in range(3)}
        }
    }

def initialize_model():
    global model, scaler, selector, features, metrics
    print("Cargando y preparando datos...")
    encuesta = cargar_datos('BD/Evaluacion_Estres_Estudiantil.csv')
    X, y, scaler, selector, features = preprocesar_datos(encuesta)
    
    print("Entrenando modelo...")
    model, X_test, y_test = entrenar_modelo(X, y)
    
    print("Calculando métricas...")
    metrics = get_model_metrics(model, X_test, y_test)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/metrics')
def get_metrics():
    return jsonify(metrics)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Crear DataFrame con los datos
        entrada = pd.DataFrame([{
            'Semestre': int(data['semestre']),
            'Horas_estudio': float(data['horasEstudio']),
            'Horas_sueno': float(data['horasSueno']),
            'Horas_redes': float(data['horasRedes']),
            'Cafeina': int(data['cafeina']),
            'Promedio': float(data['promedio'])
        }])

        # Preprocesar datos
        entrada_scaled = scaler.transform(entrada[features])
        entrada_selected = selector.transform(entrada_scaled)

        # Realizar predicción
        prediccion = model.predict(entrada_selected)[0]
        probas = model.predict_proba(entrada_selected)[0]

        # Preparar respuesta
        niveles = ['Bajo', 'Medio', 'Alto']
        response = {
            'level': niveles[prediccion],
            'probabilities': {
                'Bajo': float(probas[0]),
                'Medio': float(probas[1]),
                'Alto': float(probas[2])
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    initialize_model()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port) 