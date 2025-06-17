import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
np.random.seed(42)
returns = pd.DataFrame({
    'A': np.random.normal(0.01, 0.05, 35),
    'B': np.random.normal(0.005, 0.02, 35)
})

mean_returns = returns.mean()
cov_matrix = returns.cov()

results = []

# Генерация портфелей
for weight_a in np.arange(0.1, 1.0, 0.1):
    weight_b = 1 - weight_a
    weights = np.array([weight_a, weight_b])
    
    port_return = np.dot(weights, mean_returns)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_std = np.sqrt(port_variance)
    
    results.append({
        'Weight_A': weight_a,
        'Weight_B': weight_b,
        'Portfolio_Return': port_return,
        'Portfolio_Std': port_std
    })

# Сохраним в CSV
results_df = pd.DataFrame(results)
results_df.to_csv('portfolio_results.csv', index=False)

# Сохраним в Excel
results_df.to_excel('portfolio_results.xlsx', index=False)

# Построим график
plt.figure(figsize=(8,6))
plt.plot(results_df['Portfolio_Std'], results_df['Portfolio_Return'], marker='o')

# Добавим подписи к точкам
for i, row in results_df.iterrows():
    plt.annotate(f"A:{row['Weight_A']:.1f}\nB:{row['Weight_B']:.1f}",
                 (row['Portfolio_Std'], row['Portfolio_Return']),
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier with Weights')
plt.grid(True)

# Сохраним график
plt.savefig('efficient_frontier.png')

plt.show()
