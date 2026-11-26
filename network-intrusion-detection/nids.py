# Détection d'intrusions réseau (NIDS) par machine learning — KDD Cup 99 (10 %)
# Lance-le avec : python nids.py   (le jeu de données est téléchargé par scikit-learn au premier lancement)

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------
# Tâche 1 : charger, étiqueter, supprimer les doublons
# ---------------------------------------------------------------
df = fetch_kddcup99(percent10=True, as_frame=True).frame
df['labels'] = df['labels'].str.decode('utf-8').str.rstrip('.')
for col in ['protocol_type', 'service', 'flag']:
    df[col] = df[col].str.decode('utf-8')
num_cols = [c for c in df.columns if c not in ('protocol_type', 'service', 'flag', 'labels')]
df[num_cols] = df[num_cols].astype(float)

print("Connexions au départ :", len(df))
df = df.drop_duplicates()
print("Après suppression des doublons :", len(df))

# Les 22 attaques sont regroupées en 4 familles
FAMILLES = {
    'DoS':   ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop'],
    'Probe': ['ipsweep', 'nmap', 'portsweep', 'satan'],
    'R2L':   ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster'],
    'U2R':   ['buffer_overflow', 'loadmodule', 'perl', 'rootkit'],
}
famille = {a: f for f, attaques in FAMILLES.items() for a in attaques}
df['famille'] = df['labels'].map(famille).fillna('normal')
df['attaque'] = (df['famille'] != 'normal').astype(int)
print(df['famille'].value_counts(), "\n")

X = df.drop(columns=['labels', 'famille', 'attaque'])
y = df['attaque']

def preparation():
    # Variables texte -> one-hot ; variables numériques -> centrées-réduites
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['protocol_type', 'service', 'flag']),
        ('num', StandardScaler(), num_cols),
    ])

def taux(y_vrai, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_vrai, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn), fp / (fp + tn), tn, fp, fn, tp

# ---------------------------------------------------------------
# Tâche 2 : détection supervisée (normal / attaque)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test, fam_train, fam_test = train_test_split(
    X, y, df['famille'], test_size=0.2, random_state=42, stratify=df['famille'])

modeles = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for nom, modele in modeles.items():
    scores = cross_val_score(make_pipeline(preparation(), modele), X_train, y_train, cv=cv, scoring='f1')
    print(f"{nom} : F1 moyen en validation croisée 5 plis = {scores.mean():.4f} (+/- {scores.std():.4f})")

rf = make_pipeline(preparation(), RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
det, fpr, tn, fp, fn, tp = taux(y_test, pred)
print(f"\nRandom Forest sur le test ({len(y_test)} connexions) :")
print(classification_report(y_test, pred, target_names=['Normal', 'Attaque'], digits=4))
print(f"TN = {tn}  FP = {fp}  FN = {fn}  TP = {tp}")
print(f"Taux de détection : {det:.2%}   Taux de faux positifs : {fpr:.2%}")
print("Détection par famille :")
for f in ['DoS', 'Probe', 'R2L', 'U2R']:
    m = (fam_test == f).values
    print(f"  {f:6s}: {pred[m].mean():.1%} détectées ({m.sum()} connexions)")

# ---------------------------------------------------------------
# Tâche 3 : attaque jamais vue. On retire une famille entière de l'entraînement,
# puis on regarde si le modèle la détecte quand même au test.
# ---------------------------------------------------------------
print("\n--- Attaques inconnues (famille absente de l'entraînement) ---")
for f in ['DoS', 'Probe', 'R2L', 'U2R']:
    garde = (fam_train != f).values
    modele = make_pipeline(preparation(), RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
    modele.fit(X_train[garde], y_train[garde])
    m = (fam_test == f).values
    print(f"  {f:6s} jamais vue : {modele.predict(X_test[m]).mean():.1%} détectées")

# ---------------------------------------------------------------
# Tâche 4 : détection d'anomalies non supervisée.
# Isolation Forest apprend UNIQUEMENT le trafic normal ; tout ce qui s'en écarte est suspect.
# Le seuil est fixé pour n'avoir que 1 % de faux positifs sur du trafic normal de validation.
# ---------------------------------------------------------------
print("\n--- Isolation Forest (appris sur le trafic normal seulement) ---")
normal_train = X_train[(y_train == 0).values]
n_fit, n_val = train_test_split(normal_train, test_size=0.2, random_state=42)
prep = preparation().fit(n_fit)
iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(prep.transform(n_fit))
seuil = np.percentile(-iso.score_samples(prep.transform(n_val)), 99)   # 1 % de faux positifs visés
score_test = -iso.score_samples(prep.transform(X_test))
pred_iso = (score_test > seuil).astype(int)
det, fpr, *_ = taux(y_test, pred_iso)
print(f"Taux de détection : {det:.2%}   Taux de faux positifs : {fpr:.2%}")
for f in ['DoS', 'Probe', 'R2L', 'U2R']:
    m = (fam_test == f).values
    print(f"  {f:6s}: {pred_iso[m].mean():.1%} détectées")
