import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Twoja oryginalna implementacja regresji logistycznej
class LogisticRegression:
 
    def __init__(self, L=0.001, n_iters=10000): 
        self.L = L  # Learning rate  
        self.n_iters = n_iters 
        self.m_now = None  # Weights  
        self.b_now = None  # Bias  
 
    def fit(self, X, Y): 
        n_samples, n_features = X.shape 
        self.m_now = np.zeros(n_features) 
        self.b_now = 0 
 
        for _ in range(self.n_iters): 
            # y = mx + b 
            linear_model = np.dot(X, self.m_now) + self.b_now 
            y_predcited = self._sigmoid(linear_model) 
 
            # Gradient of cross-entropy loss for weights and bias 
            m_gradient = -(1 / n_samples) * np.dot(X.T, (Y - y_predcited))  
            b_gradient = -(1 / n_samples) * np.sum(Y - y_predcited) 
 
            self.m_now -= self.L * m_gradient 
            self.b_now -= self.L * b_gradient 
 
    def predict(self, X): 
        # y = mx + b 
        linear_model = np.dot(X, self.m_now) + self.b_now 
        y_predcited = self._sigmoid(linear_model) 
        y_predcited_class = [1 if i >= 0.5 else 0 for i in y_predcited] 
        return np.array(y_predcited_class) 
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.m_now) + self.b_now 
        return self._sigmoid(linear_model)
 
    def _sigmoid(self, x): 
        return 1 / (1 + np.exp(-x))

# Wczytanie zbioru danych Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dla uproszczenia, zróbmy klasyfikację binarną: Setosa (0) vs nie-Setosa (1)
y_binary = (y > 0).astype(int)

# Wybierzmy jedną cechę - długość płatka (3. kolumna)
feature_idx = 2  # indeks 2 to długość płatka (petal length)
X_single_feature = X[:, feature_idx].reshape(-1, 1)  # Reshaping do formatu (n_samples, 1)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X_single_feature, y_binary, test_size=0.3, random_state=42
)

# Standaryzacja cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Trenowanie modelu
model = LogisticRegression(L=0.01, n_iters=5000)
model.fit(X_train_scaled, y_train)

# Ewaluacja modelu
y_pred = model.predict(X_test_scaled)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Dokładność: {accuracy:.4f}")

# Tworzenie danych do wizualizacji funkcji sigmoidalnej
x_range = np.linspace(X_single_feature.min() - 1, X_single_feature.max() + 1, 300)
x_range_scaled = scaler.transform(x_range.reshape(-1, 1))
sigmoid_values = model.predict_proba(x_range_scaled)

# Obliczanie wartości funkcji sigmoidalnej dla punktów treningowych
y_proba_train = model.predict_proba(X_train_scaled)
y_proba_test = model.predict_proba(X_test_scaled)

# Wizualizacja
plt.figure(figsize=(12, 6))

# Wykres funkcji sigmoidalnej
plt.plot(x_range, sigmoid_values, 'b-', linewidth=2, label='Funkcja sigmoidalna')
plt.axhline(y=0.5, color='r', linestyle='--', label='Próg decyzyjny (0.5)')

# Dodanie punktów danych
plt.scatter(X_train[y_train == 0], y_proba_train[y_train == 0], 
           c='red', marker='o', label='Klasa 0 (trening)', alpha=0.7)
plt.scatter(X_train[y_train == 1], y_proba_train[y_train == 1], 
           c='blue', marker='o', label='Klasa 1 (trening)', alpha=0.7)
plt.scatter(X_test[y_test == 0], y_proba_test[y_test == 0], 
           c='red', marker='x', label='Klasa 0 (test)', alpha=0.7)
plt.scatter(X_test[y_test == 1], y_proba_test[y_test == 1], 
           c='blue', marker='x', label='Klasa 1 (test)', alpha=0.7)

# Opisanie wykresu
plt.title(f'Model regresji logistycznej na danych Iris - Cecha: {iris.feature_names[feature_idx]}')
plt.xlabel(f'{iris.feature_names[feature_idx]} [cm]')
plt.ylabel('Prawdopodobieństwo klasy 1 (nie-Setosa)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()