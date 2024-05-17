# gpu_pytorch_script.py
import torch
import torch.nn as nn
import torch.optim as optim

# Sample data
texts = ["Hello HPC with GPU", "Let's run some NLP tasks", "This is a sample text"]
labels = [0, 1, 0]  # Simple binary labels for demonstration

# Basic tokenization and vectorization
vocab = set(" ".join(texts).split())
vocab = {word: i for i, word in enumerate(vocab)}


def vectorize(text):
    vector = torch.zeros(len(vocab))
    for word in text.split():
        if word in vocab:
            vector[vocab[word]] += 1
    return vector


X = torch.stack([vectorize(text) for text in texts])
y = torch.tensor(labels, dtype=torch.long)

# Split data into train and test sets
X_train, X_test = X[:2], X[2:]
y_train, y_test = y[:2], y[2:]


# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 10
num_classes = 2
num_epochs = 5
learning_rate = 0.001

# Model, loss function, and optimizer
model = SimpleNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Inference
with torch.no_grad():
    model.eval()
    predictions = model(X_test)
    predicted_classes = torch.argmax(predictions, dim=1)
    print("Predicted classes:", predicted_classes)
    print("Actual classes:", y_test)
