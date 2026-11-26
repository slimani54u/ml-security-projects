from sklearn.model_selection import train_test_split
import pandas as pd

#Tâche 1

# Charger le fichier
data = pd.read_csv('phishing-dataset.csv', sep=';')

# Conversion des colonnes en numériques, sauf la classe
columns = data.columns[:-1]  # Exclure la dernière colonne (Class)
for col in columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Conversion de la colonne 'Class' (dernière colonne)
data['Class'] = data['Class'].apply(lambda x: 1 if x.strip().lower() == 'phishing' else 0)

# Définir X (caractéristiques) et y (labels)
X = data.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
y = data['Class']      # Dernière colonne


# Vérifier un aperçu des données
#print(data.head())

# Découper le jeu de données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vérifier les dimensions
print("Taille de l'ensemble d'apprentissage :", X_train.shape, y_train.shape)
print("Taille de l'ensemble de test :", X_test.shape, y_test.shape)


#Tâche 2

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Initialisation des modèles
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
tree_model = DecisionTreeClassifier(random_state=42)
svc_model = LinearSVC(random_state=42, max_iter=1000)

# Entraîner les modèles
logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
svc_model.fit(X_train, y_train)

# Tester les modèles
logistic_predictions = logistic_model.predict(X_test)
tree_predictions = tree_model.predict(X_test)
svc_predictions = svc_model.predict(X_test)

# Générer les rapports de classification
print("Logistic Regression:\n", classification_report(y_test, logistic_predictions, target_names=["Legitimate", "Phishing"]))
print("Decision Tree:\n", classification_report(y_test, tree_predictions, target_names=["Legitimate", "Phishing"]))
print("Linear SVC:\n", classification_report(y_test, svc_predictions, target_names=["Legitimate", "Phishing"]))

# Résultats pour Logistic Regression
# - Accuracy : 67% des prédictions sont correctes.
# - Précision pour "Phishing" : 72%, ce qui signifie que 72% des sites identifiés comme phishing sont réellement des sites phishing.
# - Rappel pour "Phishing" : 67%, ce qui indique que le modèle a détecté 67% des sites phishing parmi tous les phishing présents dans les données.
# - F1-score pour "Phishing" : 70%, une bonne balance entre précision et rappel.
# - Ce modèle a une performance moyenne, mais il manque certains sites de phishing, ce qui pourrait poser problème dans un contexte réel.


# Résultats pour Decision Tree Classifier
# - Accuracy : 71%, le meilleur parmi les trois modèles.
# - Précision pour "Phishing" : 75%, le modèle est très précis pour identifier les sites phishing.
# - Rappel pour "Phishing" : 73%, ce qui montre que le modèle détecte la majorité des sites phishing.
# - F1-score pour "Phishing" : 74%, un équilibre optimal entre précision et rappel.
# - Decision Tree est clairement le modèle le plus performant sur ces données. Il est capable de bien classifier les sites phishing et légitimes.


# Résultats pour Linear SVC
# - Accuracy : 67%, similaire à Logistic Regression.
# - Précision pour "Phishing" : 72%, comme Logistic Regression, le modèle est relativement précis.
# - Rappel pour "Phishing" : 68%, légèrement meilleur que Logistic Regression.
# - F1-score pour "Phishing" : 70%, identique à Logistic Regression.
# - Linear SVC a des performances similaires à la régression logistique. Il n'apporte pas d'amélioration notable par rapport aux deux autres modèles.


# Conclusion générale sur les résultats
# - Decision Tree Classifier est le modèle le plus performant avec une accuracy de 71% et un bon équilibre entre précision et rappel.
# - Logistic Regression et Linear SVC sont comparables, mais leurs performances sont inférieures à l'arbre de décision.
# - Pour détecter des sites phishing avec un bon niveau de fiabilité, Decision Tree est recommandé pour cette tâche.


#Tâche 3

from sklearn.model_selection import cross_val_score

# Validation croisée sur 10 plis pour Logistic Regression
logistic_cv = cross_val_score(logistic_model, X, y, cv=10, scoring='accuracy')
print("Logistic Regression - Accuracy moyenne :", logistic_cv.mean())

# Résultats pour Logistic Regression avec validation croisée
# - Accuracy moyenne : 66.2%.
# - Cela signifie que, en moyenne, 66.2% des prédictions faites par le modèle sont correctes.
# - Ce score est cohérent avec les résultats obtenus lors de la Tâche 2 (67%), ce qui montre que le modèle a une performance stable.
# - La validation croisée confirme que Logistic Regression a des performances moyennes sur ce jeu de données.

# Validation croisée sur 10 plis pour Decision Tree
tree_cv = cross_val_score(tree_model, X, y, cv=10, scoring='accuracy')
print("Decision Tree - Accuracy moyenne :", tree_cv.mean())

# Résultats pour Decision Tree avec validation croisée
# - Accuracy moyenne : 70.8%.
# - Ce modèle est le plus performant parmi les trois, comme déjà observé dans la Tâche 2.
# - La validation croisée confirme que Decision Tree a une bonne capacité à généraliser sur des données non vues.
# - La robustesse et l'efficacité de cet algorithme en font le meilleur choix pour ce cas d'usage.


# Validation croisée sur 10 plis pour Linear SVC
svc_cv = cross_val_score(svc_model, X, y, cv=10, scoring='accuracy')
print("Linear SVC - Accuracy moyenne :", svc_cv.mean())


# Résultats pour Linear SVC avec validation croisée
# - Accuracy moyenne : 66.2%.
# - Les performances de Linear SVC sont très similaires à celles de Logistic Regression.
# - Ce modèle ne semble pas tirer profit des caractéristiques du jeu de données autant que Decision Tree.
# - La validation croisée montre que Linear SVC a des performances stables mais inférieures à celles de Decision Tree.


#Tâche 4

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve

# Configuration des hyperparamètres
param_ranges = {
    "max_depth": np.arange(100, 1100, 100),  # Valeurs de 100 à 1000
    "max_leaf_nodes": np.arange(100, 1100, 100),  # Valeurs de 100 à 1000
    "criterion": ["gini", "entropy", "log_loss"],  # Critères disponibles
    "splitter": ["best", "random"]  # Splitters disponibles
}

# Préparer les sous-graphiques
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Étudier max_depth
param_range = param_ranges["max_depth"]
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(random_state=0), X, y,
    param_name="max_depth", param_range=param_range, cv=10, scoring="accuracy"
)
axes[0, 0].plot(param_range, train_scores.mean(axis=1), label="Train Score", marker='o')
axes[0, 0].plot(param_range, test_scores.mean(axis=1), label="Test Score", marker='o')
axes[0, 0].set_title("Validation Curve for max_depth")
axes[0, 0].set_xlabel("max_depth")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid()

# Étudier max_leaf_nodes
param_range = param_ranges["max_leaf_nodes"]
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(random_state=0), X, y,
    param_name="max_leaf_nodes", param_range=param_range, cv=10, scoring="accuracy"
)
axes[0, 1].plot(param_range, train_scores.mean(axis=1), label="Train Score", marker='o')
axes[0, 1].plot(param_range, test_scores.mean(axis=1), label="Test Score", marker='o')
axes[0, 1].set_title("Validation Curve for max_leaf_nodes")
axes[0, 1].set_xlabel("max_leaf_nodes")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid()

# Étudier criterion
param_range = param_ranges["criterion"]
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(random_state=0), X, y,
    param_name="criterion", param_range=param_range, cv=10, scoring="accuracy"
)
axes[1, 0].bar(param_range, test_scores.mean(axis=1), color='skyblue', edgecolor='black')
axes[1, 0].set_title("Validation Scores for criterion")
axes[1, 0].set_xlabel("criterion")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].grid(axis='y')

# Étudier splitter (amélioration pour différences au millième près)
# Étudier splitter (afficher uniquement Test Score avec une ligne horizontale)
param_range = param_ranges["splitter"]
test_scores = validation_curve(
    DecisionTreeClassifier(random_state=0), X, y,
    param_name="splitter", param_range=param_range, cv=10, scoring="accuracy"
)[1]  # Récupérer uniquement les scores de test

# Moyennes des scores
test_mean = test_scores.mean(axis=1)

# Afficher les résultats sous forme de barres
axes[1, 1].bar(param_range, test_mean, color="salmon", edgecolor="black", alpha=0.8, width=0.5, label="Test Score")

# Ajouter une ligne horizontale entre les barres
for i in range(len(param_range) - 1):
    axes[1, 1].plot(
        [i, i + 1],
        [test_mean[i], test_mean[i + 1]],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Difference" if i == 0 else "",  # Ajouter une seule légende
    )

# Ajouter les annotations pour les valeurs avec trois décimales
for i, v in enumerate(test_mean):
    axes[1, 1].text(i, v + 0.0005, f"{v:.3f}", ha='center', va='bottom', fontsize=10)

# Zoom sur l'échelle Y pour mieux voir les différences
min_y = test_mean.min() - 0.002
max_y = test_mean.max() + 0.002
axes[1, 1].set_ylim([min_y, max_y])  # Ajuster la plage de l'axe Y
axes[1, 1].set_title("Validation Scores for splitter (Only Test Score)")
axes[1, 1].set_xlabel("splitter")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].legend()
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

# Afficher les graphiques
#plt.tight_layout()
#plt.show()

# Résultats et analyse pour la Tâche 4 (Validation des hyperparamètres d'un Decision Tree)

# 1. max_depth
# - L'accuracy des scores de test reste constante pour différentes valeurs de max_depth (100 à 1000).
# - Cela montre que l'arbre atteint une profondeur suffisante pour capturer les motifs des données.
# - Conclusion : max_depth n'a pas d'impact significatif dans cette plage.

# 2. max_leaf_nodes
# - L'accuracy des scores de test augmente légèrement entre 100 et 300, puis se stabilise à partir de 300.
# - Cela indique qu'un nombre minimal de feuilles est nécessaire pour améliorer la généralisation.
# - Conclusion : Ce paramètre influence les performances dans la plage basse (100 à 300), mais devient moins important pour des valeurs plus élevées.

# 3. criterion
# - Les critères gini, entropy et log_loss donnent des performances similaires avec des accuracies proches (~0.709).
# - Cela indique que le choix du critère n'a pas d'impact significatif sur ce jeu de données.
# - Conclusion : Tous les critères testés sont interchangeables pour ce problème.

# 4. splitter
# - Une légère différence est visible entre les scores de test pour splitter='best' et splitter='random'.
# - Le splitter 'best' est légèrement meilleur, car il choisit la meilleure division à chaque étape.
# - Conclusion : Ce paramètre a un impact léger, mais il est conseillé d'utiliser splitter='best'.

# Synthèse :
# - Les paramètres qui influencent le plus les performances sont :
#   1. max_leaf_nodes (impact significatif entre 100 et 300).
#   2. splitter (préférer 'best').
# - Les paramètres max_depth et criterion ont un impact négligeable dans cette étude.

# Paramétrage recommande     :

# - max_leaf_nodes : 200 à 300.
# - splitter : best.

#Tâche 5
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Définir la grille des hyperparamètres (seulement splitter et max_leaf_nodes)
param_grid = {
    "max_leaf_nodes": list(range(200, 301, 10)),  # Plage identifiée comme optimale
    "splitter": ["best", "random"]  # Tester les deux valeurs
}

# Initialiser le modèle
decision_tree = DecisionTreeClassifier(random_state=0)

# Configurer GridSearchCV
grid_search = GridSearchCV(
    estimator=decision_tree,
    param_grid=param_grid,
    cv=10,              # Validation croisée à 10 plis
    scoring="accuracy",  # Mesurer l'accuracy
    n_jobs=-1            # Utiliser tous les cœurs pour accélérer
)

# Exécuter GridSearchCV sur les données d'entraînement
grid_search.fit(X, y)

# Afficher les meilleurs hyperparamètres et le score
print("Meilleurs hyperparamètres :", grid_search.best_params_)
print("Meilleure accuracy (validation croisée) :", grid_search.best_score_)


# 1. Meilleurs hyperparamètres trouvés :
# - max_leaf_nodes = 220 : Cette valeur optimise le compromis entre la complexité de l'arbre et sa capacité
#   à généraliser sur les données de test.
# - splitter = 'best' : Ce paramètre choisit les divisions optimales à chaque étape, améliorant la précision
#   globale.

# 2. Meilleure accuracy (validation croisée) :
# - La meilleure accuracy moyenne obtenue est de 70.79% (0.7079).
# - Cela correspond à l'évaluation sur 10 plis en validation croisée.

# 3. Comparaison avec d'autres configurations :
# - Pour max_leaf_nodes = 220, splitter='best' donne de meilleures performances que splitter='random'.
# - Cela confirme que la sélection de divisions optimales ('best') est préférable dans ce cas précis.

# 4. Conclusion :
# - Le paramètre splitter='best' est recommandé pour sa capacité à choisir des divisions optimales.
# - max_leaf_nodes = 220 offre un bon équilibre entre complexité du modèle et généralisation.
# - Ce paramétrage permet d'atteindre une accuracy moyenne de 70.79%, ce qui est légèrement meilleur
#   que d'autres combinaisons testées.

# Paramétrage final suggéré :
# - max_leaf_nodes = 220
# - splitter = 'best'



#Tâche 6
import numpy as np

# Réentraîner le modèle optimisé sur l'ensemble complet des données
optimized_tree = DecisionTreeClassifier(
    splitter='random', max_leaf_nodes=200, random_state=0
)
optimized_tree.fit(X, y)  # Ajustement sur toutes les données

# Préparer les exemples à tester
examples = np.array([
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],  # Exemple 1 : Legitimate
    [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]   # Exemple 2 : Phishing
])

# Classes attendues pour vérification
expected_classes = ["Legitimate", "Phishing"]

# Faire des prédictions
predictions = optimized_tree.predict(examples)

# Afficher les résultats
for i, prediction in enumerate(predictions):
    print(f"Exemple {i + 1}:")
    print(f"  Données : {examples[i]}")
    print(f"  Classe attendue : {expected_classes[i]}")
    print(f"  Classe prédite : {'Legitimate' if prediction == 0 else 'Phishing'}")
    print()

#Tâche 7

