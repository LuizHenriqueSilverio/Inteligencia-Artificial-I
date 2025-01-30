import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
data = pd.read_csv('Obesity prediction.csv')

# Verificar valores ausentes
print("Valores ausentes por coluna:")
print(data.isnull().sum())

# Preenchendo valores ausentes (se necessário)
data = data.dropna()  # Remover linhas com valores ausentes (ajustável)

# Codificar variáveis categóricas
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separar features e target
X = data.drop('Obesity', axis=1)
y = data['Obesity']

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Fazer previsões
y_pred = rf_model.predict(X_test)

import pickle
pickle.dump(rf_model, open('model.pkl','wb')) # salvando o modelo