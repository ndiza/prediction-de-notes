import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement des données
df = pd.read_csv("students.csv")
print("Aperçu des données :")
print(df.head())

# 2. Séparation des variables
X = df[["presence", "devoirs", "participation"]]
y = df["note_finale"]

# 3. Séparation en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Création et entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prédictions
y_pred = model.predict(X_test)

# 6. Évaluation
print("MSE :", mean_squared_error(y_test, y_pred))
print("R² Score :", r2_score(y_test, y_pred))

# 7. Affichage graphique
plt.scatter(y_test, y_pred)
plt.xlabel("Vraies notes")
plt.ylabel("Notes prédites")
plt.title("Prédiction des notes d'étudiants")
plt.grid(True)
plt.show()
