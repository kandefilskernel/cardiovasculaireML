import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Chargement des données
@st.cache_data  
def load_data():
    data = pd.read_csv('heart.csv')  # Assurez-vous que ce fichier est présent
    return data

# Prétraitement des données
def preprocess_data(data):
    # Gestion des données manquantes
    if data.isnull().sum().any():
        st.warning("Des données manquantes ont été détectées.")
        st.write(data.isnull().sum())

        # Imputation des données manquantes
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                if data[col].dtype == 'object':
                    mode = data[col].mode()[0]
                    data[col].fillna(mode, inplace=True)
                else:
                    median = data[col].median()
                    data[col].fillna(median, inplace=True)

    # Encodage des variables catégorielles
    label_encoder = LabelEncoder()
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])

    # Standardisation des variables numériques
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']  # Ajout de 'Oldpeak'
    for col in numerical_cols:
        if col in data.columns:
            # Vérification si la colonne est bien numérique
            if data[col].dtype != 'object':
                scaler = StandardScaler()
                data[col] = scaler.fit_transform(data[[col]])
            else:
                st.warning(f"La colonne {col} contient des valeurs non numériques et ne peut pas être normalisée.")

    return data

# Modèle de prédiction
def train_model(X_train, y_train, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression()
    else:
        model = SVC(probability=True)  # Ajout de SVC comme option de modèle

    model.fit(X_train, y_train)  # Entraînement du modèle
    return model

# Interface utilisateur
def main():
    st.title("Application de Prédiction de Maladies Cardiaques")

    # Chargement des données
    data = load_data()
    st.write("Données chargées :")
    st.write(data.head())

    # Prétraitement des données
    processed_data = preprocess_data(data)
    st.write("Données prétraitées :")
    st.write(processed_data.head())

    # Vérification de la colonne cible
    if 'HeartDisease' not in processed_data.columns:
        st.error("La colonne 'HeartDisease' n'est pas présente dans les données.")
        return

    # Séparation des caractéristiques et de la cible
    X = processed_data.drop('HeartDisease', axis=1)
    y = processed_data['HeartDisease']

    # Séparation des ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = train_model(X_train, y_train)

    # Prédictions
    predictions = model.predict(X_test)
    st.write("Prédictions :")
    st.write(predictions)

    # Évaluation du modèle
    st.write("Matrice de confusion :")
    cm = confusion_matrix(y_test, predictions)
    st.write(cm)

    st.write("Rapport de classification :")
    report = classification_report(y_test, predictions)
    st.text(report)

if __name__ == "__main__":
    main()
