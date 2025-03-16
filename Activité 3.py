# Partie 1: Modélisation et compréhension mathématique du problème

# 1. Écrire l’équation différentielle avec les valeurs du problème
"""
L'équation logistique est donnée par :
dP/dt = r * P * (1 - P/K)

Avec les valeurs du problème :
- r = 0.05 (taux de croissance)
- K = 100000 (capacité maximale du marché)
- P0 = 1000 (nombre initial de clients)

L'équation différentielle devient :
dP/dt = 0.05 * P * (1 - P/100000)
"""

# 2. Expliquer pourquoi l’interprétation fonctionne au début de l’activité et à la fin de l’activité
"""
- Au début de l’activité : Le nombre de clients P est très petit par rapport à K, donc P/K ≈ 0.
  L'équation se simplifie en dP/dt ≈ rP, ce qui correspond à une croissance exponentielle.
  Cela reflète une croissance rapide au début.

- À la fin de l’activité : Le nombre de clients P se rapproche de K, donc P/K ≈ 1.
  L'équation se simplifie en dP/dt ≈ 0, ce qui signifie que la croissance ralentit et se stabilise autour de la capacité maximale K.
"""

# 3. Utiliser la méthode de discrétisation pour obtenir l’équation d’Euler
"""
La méthode d'Euler consiste à approximer la solution de l'équation différentielle en discrétisant le temps.
Pour un pas de temps Δt, l'équation d'Euler est :
P_{n+1} = P_n + Δt * r * P_n * (1 - P_n / K)
"""

# 4. Implémenter en Python cette équation en choisissant un pas approprié
import numpy as np
import matplotlib.pyplot as plt

# Paramètres
r = 0.05
K = 100000
P0 = 1000
dt = 0.1  # Pas de temps
t_max = 100  # Temps maximal
n_steps = int(t_max / dt)

# Initialisation
P = np.zeros(n_steps)
P[0] = P0

# Méthode d'Euler
for i in range(1, n_steps):
    dP = r * P[i-1] * (1 - P[i-1] / K)
    P[i] = P[i-1] + dt * dP

# Affichage
time = np.arange(0, t_max, dt)
plt.plot(time, P, label="Approximation d'Euler")
plt.xlabel('Temps (mois)')
plt.ylabel('Nombre de clients')
plt.title('Croissance des clients selon le modèle logistique')
plt.legend()
plt.grid()
plt.show()


# Partie 2: Étude de la précision de la solution

# 1. Expliquer la cohérence de cette solution avec le lancement exponentiel et la saturation
"""
La solution analytique de l'équation logistique est :
P(t) = K / (1 + ((K - P0) / P0) * e^{-rt})

- Au début : t est petit, donc e^{-rt} ≈ 1, et P(t) ≈ K / (1 + (K - P0)/P0) ≈ P0 * e^{rt}, ce qui correspond à une croissance exponentielle.
- À la fin : t est grand, donc e^{-rt} ≈ 0, et P(t) ≈ K, ce qui correspond à la saturation.
"""

# 2. Implémenter cette fonction
def logistic_growth(t, r, K, P0):
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

# Calcul de la solution analytique
P_analytical = logistic_growth(time, r, K, P0)

# Affichage
plt.plot(time, P, label="Approximation d'Euler")
plt.plot(time, P_analytical, label="Solution analytique", linestyle='--')
plt.xlabel('Temps (mois)')
plt.ylabel('Nombre de clients')
plt.title('Comparaison entre approximation et solution analytique')
plt.legend()
plt.grid()
plt.show()

# 3. Comparer les résultats d’approximations avec le résultat de cette fonction pour plusieurs valeurs en vous basant sur la MSE
"""
La MSE (Mean Squared Error) est calculée comme suit :
MSE = (1/n) * Σ(P_i - P_analytical_i)^2
"""
mse = np.mean((P - P_analytical) ** 2)
print(f"MSE: {mse}")

# 4. Tracer les courbes de votre approximation et de la solution analytique
"""
Le tracé a déjà été fait dans la section précédente.
"""


# Partie 3: Application sur des données réelles et recalibrage du modèle

# 1. Tracer la courbe du nombre d’utilisateurs
"""
Supposons que vous avez un fichier Dataset_nombre_utilisateurs.csv contenant les données.
Voici comment vous pouvez le tracer :
"""
import pandas as pd

# Chargement des données
data = pd.read_csv('Dataset_nombre_utilisateurs.csv')
time_real = data['time']
P_real = data['users']

# Affichage
plt.plot(time_real, P_real, label="Données réelles")
plt.xlabel('Temps (jours)')
plt.ylabel('Nombre d’utilisateurs')
plt.title('Croissance des utilisateurs réels')
plt.legend()
plt.grid()
plt.show()

# 2. Quand est-ce qu’on atteint la phase de “saturation” ? Quand est-ce qu’on atteint 50% de la saturation ?
"""
- Saturation : Lorsque le nombre d'utilisateurs se stabilise autour de K.
- 50% de saturation : Lorsque le nombre d'utilisateurs atteint K/2.
"""

# 3. Calculer la MSE pour les 2 solutions sur plusieurs intervalles de temps
"""
Vous pouvez calculer la MSE pour différents intervalles de temps en utilisant la même méthode que précédemment.
"""

# 4. Tracer simultanément les 3 courbes
plt.plot(time, P, label="Approximation d'Euler")
plt.plot(time, P_analytical, label="Solution analytique", linestyle='--')
plt.plot(time_real, P_real, label="Données réelles")
plt.xlabel('Temps (jours)')
plt.ylabel('Nombre d’utilisateurs')
plt.title('Comparaison des modèles avec les données réelles')
plt.legend()
plt.grid()
plt.show()

# 5. Expliquer pour quelle(s) raison(s) les écarts sont significatifs
"""
Les écarts peuvent être dus à :
- Des hypothèses simplificatrices du modèle (par exemple, r constant).
- Des facteurs externes non pris en compte (concurrence, changements de marché, etc.).
- Des erreurs de mesure dans les données réelles.
"""

# 6. Proposer de nouvelles hypothèses pour donner une meilleure approximation
"""
Vous pourriez :
- Modifier r pour qu'il soit variable dans le temps.
- Ajouter des termes pour modéliser des événements spécifiques (campagnes marketing, etc.).
- Utiliser un modèle plus complexe comme un modèle de Gompertz.
"""

# 7. Recalculer la MSE et retracer les courbes
"""
Après avoir ajusté le modèle, recalculez la MSE et tracez les nouvelles courbes.
"""


# Partie 4: Question ouverte

# Est-ce que le business modèle est rentable ?
"""
Pour déterminer si le business modèle est rentable, vous devez calculer les revenus et les coûts sur une période donnée.
"""

# Est-ce qu’il manque des hypothèses ?
"""
Il pourrait manquer des hypothèses sur :
- Le taux de désabonnement des utilisateurs.
- Les coûts opérationnels supplémentaires (support client, maintenance, etc.).
- Les variations saisonnières dans l'acquisition des utilisateurs.
"""

# Plan de projection du chiffre d’affaires et des bénéfices
"""
Voici un exemple de calcul :
"""
# Paramètres
max_users_per_server = 2000
server_cost = 1000  # euros par mois
marketing_cost_per_user = 10  # euros
marketing_budget = 50000  # euros
subscription_price = 11.99  # euros par mois
market_size = 400000
market_share = 0.20
initial_users = 500

# Calcul du nombre d'utilisateurs
K = market_size * market_share
r = 0.05  # Croissance initiale moyennement rapide
P0 = initial_users

# Simulation sur 12 mois
months = 12
time_months = np.arange(0, months, 1)
P_users = logistic_growth(time_months, r, K, P0)

# Calcul des coûts et revenus
servers_needed = np.ceil(P_users / max_users_per_server)
server_costs = servers_needed * server_cost
marketing_costs = np.where(time_months == 0, marketing_budget * 0.35, marketing_budget * 0.65 / (months - 1)) + marketing_cost_per_user * P_users
revenues = P_users * subscription_price
profits = revenues - server_costs - marketing_costs

# Affichage
plt.plot(time_months, revenues, label="Revenus")
plt.plot(time_months, profits, label="Bénéfices")
plt.xlabel('Mois')
plt.ylabel('Euros')
plt.title('Projection des revenus et bénéfices sur 12 mois')
plt.legend()
plt.grid()
plt.show()