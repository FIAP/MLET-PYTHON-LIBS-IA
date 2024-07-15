import torch

# Verificar se há uma GPU disponível e definir o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Criar uma matriz aleatória grande
matrix_size = 1000
matrix = torch.randn(matrix_size, matrix_size, device=device)
print("Matriz criada e movida para a GPU.")

# Calcular a decomposição de valores singulares (SVD)
U, S, V = torch.svd(matrix)
print("Decomposição de valores singulares (SVD) calculada na GPU.")

# Exibir os resultados
print("Matriz U:")
print(U)
print("Matriz S (valores singulares):")
print(S)
print("Matriz V:")
print(V)
