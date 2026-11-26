# TP Spams : détection de spams dans des mails par machine learning
# Lance-le avec : python TpSpams.py  (depuis ce dossier)

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Tâche 1 : charger et nettoyer les données
data = pd.read_csv('spam-dataset.csv')
data = data.dropna()
print("Mails au départ :", len(data))

# Piège n° 1 : les doublons. Sans ça, le même mail peut se retrouver
# dans l'entraînement ET dans le test, et le score est gonflé.
data = data.drop_duplicates(subset='email')
print("Mails après suppression des doublons :", len(data))
print("Répartition (0 = légitime, 1 = spam) :\n", data['label'].value_counts())

X = data['email']
y = data['label']

# 80 % entraînement / 20 % test, en gardant la même part de spams (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Tâche 2 : comparer 3 modèles. Le texte est transformé en nombres avec TF-IDF
# (poids de chaque mot et paire de mots). Le pipeline refait le TF-IDF
# à chaque pli de la validation croisée : pas de fuite d'information.
modeles = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVC": LinearSVC(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultats = {}
for nom, modele in modeles.items():
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), modele)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')
    resultats[nom] = scores.mean()
    print(f"{nom} : F1 moyen en validation croisée 5 plis = {scores.mean():.3f} (+/- {scores.std():.3f})")

# Tâche 3 : évaluer le meilleur modèle une seule fois sur le jeu de test
meilleur = max(resultats, key=resultats.get)
print("\nMeilleur modèle :", meilleur)
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), modeles[meilleur])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print(classification_report(y_test, pred, target_names=["Legitimate", "Spam"], digits=3))
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print("Matrice de confusion : TN =", tn, " FP =", fp, " FN =", fn, " TP =", tp)
print(f"Taux de détection (rappel spam) : {tp / (tp + fn):.1%}")
print(f"Précision spam : {tp / (tp + fp):.1%}")
print(f"Taux de faux positifs (vrais mails bloqués) : {fp / (fp + tn):.1%}")

# ROC-AUC : qualité du classement, indépendante du seuil
score = pipe.decision_function(X_test) if hasattr(pipe, "decision_function") else pipe.predict_proba(X_test)[:, 1]
print(f"ROC-AUC : {roc_auc_score(y_test, score):.3f}")

# Tâche 4 : les 15 mots qui pèsent le plus vers « spam » (explicabilité)
vec = pipe.named_steps['tfidfvectorizer']
clf = pipe.steps[-1][1]
if hasattr(clf, "coef_"):
    poids = clf.coef_[0]
elif hasattr(clf, "feature_log_prob_"):
    poids = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
mots = vec.get_feature_names_out()
top = sorted(zip(poids, mots), reverse=True)[:15]
print("\nMots les plus « spam » :", ", ".join(m for _, m in top))
