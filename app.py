import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Chargement des données
@st.cache
def load_data():
    data = pd.read_csv('heart.csv')
    return data

# Prétraitement des données
def preprocess_data(data):
    # Gestion des données manquantes
    if data.isnull().sum().any():
        st.warning("Des données manquantes ont été détectées.")
        # Affichage des données manquantes
        st.write(data.isnull().sum())

        # Imputation des données manquantes
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                if data[col].dtype == 'object':
                    # Imputation par le mode pour les colonnes catégorielles
                    mode = data[col].mode()[0]
                    data[col].fillna(mode, inplace=True)
                else:
                    # Imputation par la médiane pour les colonnes numériques
                    median = data[col].median()
                    data[col].fillna(median, inplace=True)

    # Encodage des variables catégorielles
    label_encoder = LabelEncoder()
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina']
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Standardisation des variables numériques
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data

# Modèle de prédiction
def train_model(X_train, y_train, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression()
    # Note: Vous devez ajouter le code pour entraîner le modèle ici
