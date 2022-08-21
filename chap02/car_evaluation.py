# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline


# %%
# dataset = pd.read_csv('./Torch/chap02/data/car_evaluation.csv')
dataset = pd.read_csv('./data/car_evaluation.csv')

# %%
print(dataset.head())
print()
print(dataset.info())
print()
print(dataset.describe())

# %%
# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 8
# fig_size[1] = 6
# plt.rcParams["figure.figsize"] = fig_size

plt.figure(figsize=(8 ,6))
dataset.output.value_counts().plot(kind='pie', 
                                   autopct='%0.02f%%', 
                                   colors=['lightblue', 'lightgreen', 'orange', 'pink'],
                                   explode=(0.05, 0.05, 0.05, 0.05))
plt.show()

# %%
categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')

# %%
# print(dataset.head())
# print()
# print(dataset.info())
# print()
# print(dataset.describe())

# %%
price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)

# %%
print(f"price.shape: {price.shape}")
print()
print(f"categorical_data.shape: {categorical_data.shape}")
print()
print(pd.DataFrame(categorical_data).head())
print()
print(pd.DataFrame(categorical_data).info())
print()
print(pd.DataFrame(categorical_data).describe())

# %%
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
print(categorical_data[:5])

# %%
outputs = pd.get_dummies(dataset.output)
outputs = outputs.values
outputs = torch.tensor(outputs).flatten()

print(f"categorical_data.shape: {categorical_data.shape}")
print(f"outputs.shape: {outputs.shape}")

# %%
categorical_column_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size + 1) // 2)) for col_size in categorical_column_sizes]

# 임베딩 크기에 대한 정확한 규칙은 없지만, 칼럼의 고유 값 수를 2로 나누는 것을 많이 사용합니다.
# (모든 범주형 칼럼의 고유 값 수, 차원의 크기)
# TODO 임베딩과 차원의 크기
print(categorical_embedding_sizes)

# %%
total_records = 1728
test_records = int(total_records * 0.2)

categorical_train_data = categorical_data[:(total_records - test_records)]
categorical_test_data = categorical_data[(total_records - test_records):total_records]
train_outputs = outputs[:(total_records - test_records)]
test_outputs = outputs[(total_records - test_records):total_records]

# %%
print(f"categorical train: {len(categorical_train_data)}")
print(f"outputs train    : {len(train_outputs)}")
print(f"categorical test : {len(categorical_test_data)}")
print(f"outputs test     : {len(test_outputs)}")

# %%
class Model(nn.Module):
    def __init__(self, embedding_size, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)

        # TODO layer 만드는 방법
        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols 

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x

# %%
model = Model(categorical_embedding_sizes, 4, [200,100,50], p=0.4)
print(model)

# %%
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
if torch.backends.mps.is_available():
    print("device: mps")
    device = torch.device('mps')
else:
    print("device: cpu")
    device = torch.device('cpu')

# %%
epochs = 500
aggregated_losses = []
train_outputs = train_outputs.to(device=device, dtype=torch.int64)
for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data)
    y_pred = y_pred.to(device=device)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.2f}')
# %%
