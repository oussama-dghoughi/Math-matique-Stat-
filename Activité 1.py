
import numpy as np  
import matplotlib.pyplot as plt  

# Définition de la fonction pour calculer le coût global
def cout_global(cout_fixe, cout_unitaire, nb_produits):
    """
    Calcule le coût global en fonction des coûts fixes, du coût unitaire et du nombre de produits.
    
    :param cout_fixe: Coût fixe (coût indépendant du nombre de produits)
    :param cout_unitaire: Coût unitaire (coût par produit)
    :param nb_produits: Nombre de produits
    :return: Coût global
    """
    return cout_fixe + cout_unitaire * nb_produits

# Paramètres de base pour les calculs
cout_fixe_base = 1000  # Coût fixe de base
cout_unitaire_base = 50  # Coût unitaire de base
nb_produits_max = 100  # Nombre maximum de produits pour les calculs

# 1. Variation du coût fixe
cout_fixes = [500, 1000, 1500]  # Différentes valeurs de coûts fixes à tester

# Boucle pour tracer les courbes pour chaque coût fixe
for cf in cout_fixes:
    nb_produits = np.arange(0, nb_produits_max + 1)  # Création d'un tableau de 0 à nb_produits_max
    couts = cout_global(cf, cout_unitaire_base, nb_produits)  # Calcul des coûts globaux
    plt.plot(nb_produits, couts, label=f'Coût fixe = {cf}')  # Tracé de la courbe

#
plt.title("Variation du coût fixe")
plt.xlabel("Nombre de produits")
plt.ylabel("Coût global")
plt.legend()  #
plt.grid(True)  
plt.show()  

# 2. Variation du coût unitaire
cout_unitaires = [30, 50, 70]  # Différentes valeurs de coûts unitaires à tester

# Boucle pour tracer les courbes pour chaque coût unitaire
for cu in cout_unitaires:
    nb_produits = np.arange(0, nb_produits_max + 1)  # Création d'un tableau de 0 à nb_produits_max
    couts = cout_global(cout_fixe_base, cu, nb_produits)  # Calcul des coûts globaux
    plt.plot(nb_produits, couts, label=f'Coût unitaire = {cu}')  # Tracé de la courbe


plt.title("Variation du coût unitaire")
plt.xlabel("Nombre de produits")
plt.ylabel("Coût global")
plt.legend()  
plt.grid(True)  
plt.show()  

# 3. Variation du nombre de produits
nb_produits_max_list = [50, 100, 150]  # Différentes valeurs de nombre maximum de produits à tester

#
for npm in nb_produits_max_list:
    nb_produits = np.arange(0, npm + 1)  
    couts = cout_global(cout_fixe_base, cout_unitaire_base, nb_produits)  # Calcul des coûts globaux
    plt.plot(nb_produits, couts, label=f'Nb produits max = {npm}')  

plt.title("Variation du nombre de produits")
plt.xlabel("Nombre de produits")
plt.ylabel("Coût global")
plt.legend()  
plt.grid(True)  
plt.show()  # Affichage du graphique