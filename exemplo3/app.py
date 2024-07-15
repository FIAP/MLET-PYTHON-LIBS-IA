import torch
import torch.optim as optim

# Dados de entrada e saída desejada
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)
y_data = torch.tensor([[2.5], [3.5], [5.5], [6.5]], requires_grad=False)

# Definindo um modelo de rede neural simples com uma camada linear
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # Uma camada linear

    def forward(self, x):
        return self.linear(x)

# Inicializando o modelo
model = SimpleNN()

# Função de perda 
loss_function = torch.nn.MSELoss()

# Configurando o otimizador
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Treinamento do modelo
for epoch in range(100):
    # Forward pass
    y_pred = model(x_data)

    # Cálculo da perda
    loss = loss_function(y_pred, y_data)

    # Retropropagação e otimização
    optimizer.zero_grad()  # Zera os gradientes acumulados
    loss.backward()        # Calcula gradientes
    optimizer.step()        # Atualiza parâmetros

# Exibindo parâmetros finais
for name, param in model.named_parameters():
    print(f"{name}: {param.data.item()}")

# Fazer uma previsão com o modelo treinado
x_test = torch.tensor([[5.0]])
y_test_pred = model(x_test)
print(f"Previsão para x=5.0: {y_test_pred.item()}")

