import pandas as pd
import os
import pickle
import time
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import clean_data

def analyze_numerical_correlations(df):
    """Analyse les corrélations entre variables numériques"""
    # Sélectionner les colonnes numériques
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Identifier les paires avec corrélation > 0.9
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                # Calculer la corrélation moyenne de chaque variable avec toutes les autres
                avg_corr1 = corr_matrix[col1].abs().mean()
                avg_corr2 = corr_matrix[col2].abs().mean()
                high_corr_pairs.append({
                    'var1': col1,
                    'var2': col2,
                    'correlation': corr_matrix.iloc[i, j],
                    'to_drop': col1 if avg_corr1 > avg_corr2 else col2,
                    'reason': f"Plus forte corrélation moyenne avec autres variables ({col1 if avg_corr1 > avg_corr2 else col2})"
                })
    
    return corr_matrix, high_corr_pairs
    

def analyze_categorical_correlations(df):
    """Analyse les corrélations entre variables catégorielles avec chi2"""
    cat_cols = df.select_dtypes(include=['object']).columns
    chi2_results = []
    
    for i in range(len(cat_cols)):
        for j in range(i):
            try:
                contingency = pd.crosstab(df[cat_cols[i]], df[cat_cols[j]])
                chi2, p_value, _, _ = chi2_contingency(contingency)
                # Calculer V de Cramer pour normaliser
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                cramer_v = np.sqrt(chi2 / (n * min_dim))
                
                if cramer_v > 0.9:  # Forte association
                    chi2_results.append({
                        'var1': cat_cols[i],
                        'var2': cat_cols[j],
                        'cramer_v': cramer_v,
                        'p_value': p_value
                    })
            except:
                continue
    
    return chi2_results


def apply_pca(df):
    """Applique PCA sur les variables numériques"""
    # Préparer les données
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    X = df[numeric_cols]
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculer variance expliquée cumulative
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Trouver nombre de composantes pour 95% de variance
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    
    return {
        'pca': pca,
        'X_pca': X_pca,
        'cumulative_variance': cumulative_variance_ratio,
        'n_components_95': n_components_95
    }


# Fonction principale
def mainPlot(df):
    # 1. Analyse des corrélations
    print("1. Analyse des corrélations...")
    corr_matrix, high_corr_pairs = analyze_numerical_correlations(df)
    chi2_results = analyze_categorical_correlations(df)
    
    # Variables à supprimer
    to_drop = list(set([pair['to_drop'] for pair in high_corr_pairs]))
    
    # Création du nouveau dataset
    df_reduced = df.drop(columns=to_drop)
    
    # 2. PCA
    print("\n2. Application de PCA...")
    pca_results = apply_pca(df_reduced)
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_corr_pairs,
        'chi2_results': chi2_results,
        'variables_dropped': to_drop,
        'df_reduced': df_reduced,
        'pca_results': pca_results
    }

def plot_results(results):
    # Créer une figure avec deux sous-plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap des corrélations
    sns.heatmap(results['correlation_matrix'], ax=ax1, cmap='coolwarm')
    ax1.set_title('Matrice de corrélation')
    
    # Courbe de variance expliquée PCA
    pca_var = results['pca_results']['cumulative_variance']
    ax2.plot(range(1, len(pca_var) + 1), pca_var, 'bo-')
    ax2.axhline(y=0.95, color='r', linestyle='--')
    ax2.set_title('Variance expliquée cumulative (PCA)')
    ax2.set_xlabel('Nombre de composantes')
    ax2.set_ylabel('Variance expliquée cumulative')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def save_ann_metrics(metrics, file_name):
    # Créer le chemin du fichier avec le nom fourni
    file_path = file_name+'metrics.json'
    
    # Vérifie si le fichier existe, sinon il sera créé
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump(metrics, f)
        print(f"Le fichier {file_path} a été créé et les données ont été enregistrées.")
    else:
        print(f"Le fichier {file_path} existe déjà.")


def analyser_distribution(y, title="Distribution des prix"):
    """
    Analyse et affiche la distribution des prix
    """
    plt.figure(figsize=(12, 6))
    
    # Histogramme
    plt.subplot(1, 2, 1)
    sns.histplot(data=y, bins=30)
    plt.title(f"{title} - Histogramme")
    plt.xlabel("Prix")
    plt.ylabel("Fréquence")
    
    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=y)
    plt.title(f"{title} - Boxplot")
    plt.xlabel("Prix")
    
    plt.tight_layout()
    
    # Calculer les statistiques descriptives
    stats = {
        'Moyenne': np.mean(y),
        'Médiane': np.median(y),
        'Écart-type': np.std(y),
        'Skewness': pd.Series(y).skew(),
        'Kurtosis': pd.Series(y).kurtosis()
    }
    
    return stats

# def categoriser_prix(y):
#     """
#     Catégorise les prix en classes
#     """
#     # Utiliser les quantiles pour définir les classes de prix
#     q1, q2, q3 = np.percentile(y, [25, 50, 75])
    
#     categories = pd.cut(y, 
#                        bins=[0, q1, q2, q3, float('inf')],
#                        labels=['Bas', 'Moyen-bas', 'Moyen-haut', 'Élevé'])
    
#     return categories
def categoriser_prix(y):
    """
    Catégorise les prix en classes
    """
    # Utiliser les quantiles pour définir les classes de prix
    q1, q2, q3 = np.percentile(y, [25, 50, 75])
    
    categories = pd.cut(y, 
                       bins=[0, q1, q2, q3, float('inf')], 
                       labels=['Bas', 'Moyen-bas', 'Moyen-haut', 'Élevé'])
    
    # Convertir les catégories en entiers
    category_map = {'Bas': 0, 'Moyen-bas': 1, 'Moyen-haut': 2, 'Élevé': 3}
    y_int = categories.map(category_map).astype(int)
    
    return y_int

def appliquer_smote(X, y_cat):
    """
    Applique SMOTE pour rééquilibrer les données
    """
    # Encoder les catégories en valeurs numériques
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_cat)
    
    # Appliquer SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    
    # Reconvertir les labels
    y_resampled = le.inverse_transform(y_resampled)
    
    return X_resampled, y_resampled

def afficher_comparaison_distribution(y_orig, y_resampled):
    """
    Affiche la comparaison des distributions avant/après SMOTE
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribution originale
    y_orig.value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Distribution originale')
    ax1.set_ylabel('Nombre d\'observations')
    
    # Distribution après SMOTE
    pd.Series(y_resampled).value_counts().plot(kind='bar', ax=ax2)
    ax2.set_title('Distribution après SMOTE')
    ax2.set_ylabel('Nombre d\'observations')
    
    plt.tight_layout()
    return fig

def main(df):
    # 1. Analyser la distribution initiale
    stats_orig = analyser_distribution(df['SalePrice'])
    
    # 2. Catégoriser les prix
    prix_categories = categoriser_prix(df['SalePrice'])
    
    # 3. Préparer les features pour SMOTE
    X = df.drop('SalePrice', axis=1)
    
    # Convertir les variables catégorielles en numériques
    X = pd.get_dummies(X)
    
    # 4. Appliquer SMOTE
    X_resampled, y_resampled = appliquer_smote(X, prix_categories)
    
    # 5. Afficher les résultats
    print("\nStatistiques de la distribution originale:")
    for key, value in stats_orig.items():
        print(f"{key}: {value:.2f}")
    
    print("\nDistribution des classes avant rééchantillonnage:")
    print(prix_categories.value_counts())
    
    print("\nDistribution des classes après rééchantillonnage:")
    print(pd.Series(y_resampled).value_counts())
    
    return {
        'X_resampled': X_resampled,
        'y_resampled': y_resampled,
        'prix_categories': prix_categories
    }



def create_ann_model(input_dim):
    """Crée un modèle Artificial Neural Network (ANN)"""
    model = Sequential()
    
    # Première couche dense (entrée)
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    
    # Deuxième couche dense
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    # Couche de sortie
    model.add(Dense(4, activation='softmax'))  # 4 classes (Bas, Moyen-bas, Moyen-haut, Élevé)
    
    # Compilation du modèle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_ann(X_train, y_train, X_test, y_test):
    """Entraîne le modèle ANN et évalue les résultats"""
    # Créer le modèle
    model = create_ann_model(X_train.shape[1])
    
    # Entraînement du modèle
    history = model.fit(X_train, y_train, epochs=200, batch_size=50, validation_data=(X_test, y_test), verbose=1)
    
    save_ann_metrics(history.history, "house_price_model")
    
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    y_pred = tf.argmax(y_pred, axis=1).numpy()
    
    # Évaluation
    print("Classification Report :\n", classification_report(y_test, y_pred))
    print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))
    
    return model


def save_model(model, model_filename="house_price_model.pkl", max_attempts=5):
    """
    Sauvegarde un modèle en vérifiant sa création et sa lisibilité. 
    En cas d'échec, essaie de le sauvegarder dans un répertoire temporaire.

    Args:
        model (object): Le modèle à sauvegarder.
        model_filename (str): Nom du fichier du modèle (par défaut 'house_price_model.pkl').
        max_attempts (int): Nombre maximal de tentatives pour vérifier la sauvegarde du modèle.
    """
    # Afficher le répertoire de travail et ses permissions
    current_dir = os.getcwd()
    print(f"Répertoire courant : {current_dir}")
    print(f"Contenu du répertoire : {os.listdir()}")
    print(f"Permissions du répertoire : {oct(os.stat(current_dir).st_mode)[-3:]}")

    # Chemin absolu du fichier modèle
    model_path = os.path.join(current_dir, model_filename)
    print(f"Chemin complet du modèle : {model_path}")

    try:
        # Création du répertoire parent si nécessaire
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Sauvegarde du modèle avec vérification
        print("Début de la sauvegarde...")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print("Sauvegarde terminée")
        
        # Vérification et attente
        attempt = 0
        while attempt < max_attempts:
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                print(f"🎉 Modèle sauvegardé ! Taille : {file_size} bytes")
                
                # Vérifier que le fichier est lisible
                try:
                    with open(model_path, 'rb') as f:
                        test_load = pickle.load(f)
                    print("✅ Fichier vérifié et lisible")
                    break
                except Exception as e:
                    print(f"⚠️ Fichier créé mais illisible : {e}")
            
            print(f"Tentative {attempt + 1}/{max_attempts}...")
            time.sleep(2)
            attempt += 1
        else:
            print("⚠️ Échec de la sauvegarde après plusieurs tentatives")

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {str(e)}")
        # En cas d'erreur, essayer dans /tmp
        try:
            backup_path = os.path.join('/tmp', model_filename)
            print(f"Tentative de sauvegarde dans : {backup_path}")
            with open(backup_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Modèle sauvegardé dans le répertoire temporaire : {backup_path}")
        except Exception as e2:
            print(f"❌ Échec de la sauvegarde de secours : {str(e2)}")

def load_csv(paths):
    """Essaie de charger un fichier CSV à partir d'une liste de chemins."""
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                print(f"Erreur : Le fichier {path} est vide.")
            except pd.errors.ParserError:
                print(f"Erreur : Le fichier {path} est mal formaté.")
    print(f"Erreur : Aucun fichier trouvé dans {paths}.")
    return None

# Définir les chemins possibles
train_paths = ["work/datasets/train.csv", "datasets/train.csv"]
test_paths = ["work/datasets/test.csv", "datasets/test.csv"]


if __name__ == "__main__":
    
    train_df = clean_data(load_csv(train_paths))
    test_df =clean_data(load_csv(test_paths))

    resultatsPca = mainPlot(train_df)
    resultats = main(train_df)

    # 1. Utilisation des données avant SMOTE
    # X = train_df.drop('SalePrice', axis=1)
    X = resultatsPca['pca_results']['X_pca']
    y = resultats['prix_categories']

    # Convertir les variables catégorielles en numériques
    # X = pd.get_dummies(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Entraînement sur les données avant rééchantillonnage
    # print("Entraînement du modèle sur les données avant rééchantillonnage...")
    # model_before_smote = train_ann(X_train, y_train, X_test, y_test)

    # # 2. Utilisation des données après SMOTE
    X_resampled, y_resampled = appliquer_smote(X, resultats['prix_categories'])
    X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Entraînement sur les données après rééchantillonnage
    print("Entraînement du modèle sur les données après rééchantillonnage...")
    model_after_smote = train_ann(X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled)
    save_model(model_after_smote)
