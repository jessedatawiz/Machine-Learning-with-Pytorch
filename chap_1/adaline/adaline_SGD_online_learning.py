from adaline_stochastic_class import AdalineSGD
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("=== SIMULAÇÃO: SISTEMA DE RECOMENDAÇÃO ===")

# Modelo para recomendar produtos
recommender = AdalineSGD(eta=0.01)

print("Dia 1: Usuário vê 3 produtos")
dia1_X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])  # Features dos produtos
dia1_y = np.array([1, 0, 1])  # 1 = gostou, 0 = não gostou
recommender.partial_fit(dia1_X, dia1_y)
print("Modelo aprendeu preferências iniciais")

print("\nDia 2: Novas interações")
novo_produto_X = np.array([[0, 1, 1]])
novo_produto_y = np.array([1])  # Usuário gostou
recommender.partial_fit(novo_produto_X, novo_produto_y)
print("Modelo atualizado com nova preferência")

print("\nDia 3: Mais interações")
mais_produtos_X = np.array([[1, 0, 0], [0, 0, 1]])
mais_produtos_y = np.array([0, 1])  # Não gostou, gostou
recommender.partial_fit(mais_produtos_X, mais_produtos_y)
print("Modelo refinado com mais dados")

print("\nModelo agora está personalizado para o usuário!")

# Plot da curva de aprendizado
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(recommender.losses_) + 1), recommender.losses_, marker='o', linestyle='-', linewidth=2, markersize=6)
plt.xlabel('Épocas de Treinamento', fontsize=12)
plt.ylabel('Erro Quadrático Médio (MSE)', fontsize=12)
plt.title('Curva de Aprendizado do Modelo de Recomendação (Online Learning)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
print("\nGráfico salvo como 'learning_curve.png'")
# plt.show()