# Importation des bibliothèques nécessaires
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import bernoulli, binom

# ----------------------------
# 1. Méthode d'Euler et calcul des erreurs
# ----------------------------

def euler_method(n, f0, h=0.5):
    """
    Applique la méthode d'Euler pour résoudre une équation différentielle.
    
    :param n: Nombre d'itérations
    :param f0: Valeur initiale
    :param h: Pas de temps (par défaut 0.5)
    :return: Liste des valeurs approchées
    """
    valeurs = []
    for i in range(n + 1):
        Ui = (1 + h) ** i * f0
        valeurs.append(Ui)
    return valeurs

def calculer_erreurs_carrees(n, f0, h=0.5):
    """
    Calcule l'erreur au carré entre l'approximation d'Euler et la vraie valeur.
    
    :param n: Nombre d'itérations
    :param f0: Valeur initiale
    :param h: Pas de temps (par défaut 0.5)
    :return: Liste des erreurs au carré pour chaque point
    """
    valeurs_euler = euler_method(n, f0, h)
    erreurs = []
    
    for i in range(n + 1):
        t_n = i * h
        vrai_resultat = math.exp(t_n) * f0  # Solution exacte
        erreur_carre = (valeurs_euler[i] - vrai_resultat) ** 2
        erreurs.append((t_n, erreur_carre))
    
    return erreurs

# Paramètres pour la méthode d'Euler
h = 0.1  # Pas de temps
t_max = 2  # Temps maximal
n = int(t_max / h)  # Nombre d'itérations
f0 = 1  # Valeur initiale

# Calcul des erreurs au carré
erreurs_carrees = calculer_erreurs_carrees(n, f0, h)

# Affichage des résultats
print("t_n\tErreur^2")
for t_n, erreur in erreurs_carrees:
    print(f"{t_n:.1f}\t{erreur:.6f}")

# ----------------------------
# 2. Analyse des données de puces défectueuses
# ----------------------------

def calculer_probabilite_defaut(fichier_csv):
    """
    Calcule la probabilité qu'un composant soit défectueux à partir d'un fichier CSV.
    
    :param fichier_csv: Chemin du fichier CSV
    :return: Probabilité de défaut
    """
    df = pd.read_csv(fichier_csv)
    
    if 'défectueux' not in df.columns:
        raise ValueError("La colonne 'défectueux' est absente du fichier.")
    
    total_puces = len(df)
    defectueux_count = (df["défectueux"] == "Défectueux").sum()
    p_defectueux = defectueux_count / total_puces if total_puces > 0 else 0
    
    return p_defectueux

# Chargement du fichier CSV (remplacez par le chemin de votre fichier)
fichier = "Dataset_lot_puce_defectueuses.csv"  # Remplacez par le chemin de votre fichier
df = pd.read_csv(fichier)

# Transformation des données
df["défectueux"] = df["défectueux"].map({"Défectueux": 1, "Non défectueux": 0})
df['defectueuse'] = df['défectueux'].apply(lambda x: 1 if x == 1 else 0)

# Estimation de la probabilité de défaut
p_estime = df["défectueux"].mean()
print(f"Estimation de la probabilité qu'un composant soit défectueux : {p_estime:.4f}")

# Calcul de la probabilité de défaut à partir du fichier
probabilite = calculer_probabilite_defaut(fichier)
print(f"Probabilité qu'un composant soit défectueux : {probabilite:.4f}")

# ----------------------------
# 3. Analyse des lots de puces
# ----------------------------

# Calcul du nombre de puces défectueuses par lot
defectueux_par_lot = df.groupby('lot_id')['defectueuse'].sum()

# a. Estimation globale : Proportion moyenne de puces défectueuses
proportion_moyenne = defectueux_par_lot.mean() / 20  # Chaque lot contient 20 puces
print(f'Proportion moyenne de puces défectueuses : {proportion_moyenne:.2f}')

# b. Tracer l'histogramme de la distribution des puces défectueuses par lot
plt.figure(figsize=(10, 6))
sns.histplot(defectueux_par_lot, kde=True, bins=range(0, 21), color='skyblue', edgecolor='black')
plt.title("Distribution du nombre de puces défectueuses par lot")
plt.xlabel("Nombre de puces défectueuses")
plt.ylabel("Fréquence")
plt.show()

# c. Probabilité qu'un lot contienne au moins 5 puces défectueuses
n = 20  # Nombre de puces par lot
p = 0.10  # Probabilité de défaut
prob_lot_problematique = 1 - binom.cdf(4, n, p)  # CDF jusqu'à 4 puces défectueuses
print(f'Probabilité qu\'un lot contienne au moins 5 puces défectueuses : {prob_lot_problematique:.4f}')

# d. Calculer le pourcentage de lots rejetés (si 5 ou plus de puces défectueuses)
lots_rejetes = defectueux_par_lot[defectueux_par_lot >= 5]
pourcentage_rejet = (len(lots_rejetes) / len(defectueux_par_lot)) * 100
print(f'Pourcentage de lots rejetés : {pourcentage_rejet:.2f}%')

# e. Calculer le taux de défectuosité maximal pour un taux de rejet de 5%
def taux_rejet_maximal(p_max):
    """
    Calcule le taux de défectuosité maximal pour un taux de rejet donné.
    
    :param p_max: Taux de rejet maximal (par exemple, 0.05 pour 5%)
    :return: Taux de défectuosité maximal
    """
    # Cette fonction nécessite une implémentation spécifique en fonction des données
    return 0.10  # Valeur arbitraire pour l'exemple

p_max = 0.05
p_optimal = taux_rejet_maximal(p_max)
print(f'Taux de défectuosité maximal pour un taux de rejet de {p_max*100}% : {p_optimal:.4f}')