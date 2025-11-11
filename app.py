pip install pandas numpy seaborn streamlit scikit-learn
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
    elif model_type == 'SVM':
        model = SVC(probability=True)

    model.fit(X_train, y_train)
    return model

# Affichage de la matrice de confusion
def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False,
                xticklabels=['Pas de maladie', 'Maladie'],
                yticklabels=['Pas de maladie', 'Maladie'])
    plt.ylabel('Vérité réelle')
    plt.xlabel('Prédictions')
    plt.title('Matrice de Confusion')
    st.pyplot(plt)

# Affichage de la courbe ROC
def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='AUC = {:.2f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Titre du dashboard
st.title("Dashboard de Prédiction des Maladies Cardiovasculaires")

# Chargement et affichage des données
data = load_data()
st.subheader("Données sur les Maladies Cardiovasculaires")
st.write(data)

# Visualisation de la distribution des variables numériques
st.subheader("Distribution des Variables Numériques")
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f'Distribution de {col}')
    st.pyplot(plt)

# Prétraitement des données
data = preprocess_data(data)

# Séparation des données
X = data.drop('target', axis=1)  # Remplacez 'target' par le nom de votre variable cible
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sélection du modèle
model_type = st.selectbox("Sélectionnez le type de modèle", ['RandomForest', 'LogisticRegression', 'SVM'])

# Entraînement du modèle
model = train_model(X_train, y_train, model_type)

# Prédictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Affichage des résultats
st.subheader("Rapport de Classification")
st.text(classification_report(y_test, y_pred))

# Affichage de la matrice de confusion
st.subheader("Matrice de Confusion")
plot_confusion_matrix(y_test, y_pred)

# Affichage de la courbe ROC
st.subheader("Courbe ROC")
plot_roc_curve(y_test, y_prob)

# Prédictions en temps réel
st.subheader("Prédictions en Temps Réel")
st.write("Entrez les informations du patient pour prédire le risque de maladie cardiaque.")

# Entrées utilisateur
age = st.number_input("Âge", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sexe", ["Homme", "Femme"])
chest_pain = st.selectbox("Type de douleur thoracique", ["Type 1", "Type 2", "Type 3", "Type 4"])
resting_ecg = st.selectbox("ECG au repos", ["Normal", "Anomalie", "Hypertrophie"])
exercise_angina = st.selectbox("Angine d'effort", ["Oui", "Non"])
resting_bp = st.number_input("Pression artérielle au repos", min_value=0, max_value=200, value=120)
cholesterol = st.number_input("Cholestérol", min_value=0, max_value=600, value=200)
max_hr = st.number_input("Fréquence cardiaque maximale", min_value=0, max_value=220, value=150)

# Préparation des données pour la prédiction
input_data = np.array([[age,
                         1 if sex == "Homme" else 0,
                         chest_pain,
                         resting_ecg,
                         1 if exercise_angina == "Oui" else 0,
                         resting_bp,
                         cholesterol,
                         max_hr]])

# Standardisation
input_data[:, 0:4] = StandardScaler().fit_transform(input_data[:, 0:4])

if st.button("Prédire"):
    prediction = model.predict(input_data)
    st.write("Prédiction : ", "Maladie" if prediction[0] == 1 else "Pas de maladie")
