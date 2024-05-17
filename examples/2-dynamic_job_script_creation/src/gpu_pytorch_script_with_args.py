# gpu_pytorch_script.py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


def load_data(file_path):
    """Load text data from a file."""
    with open(file_path, "r") as file:
        data = file.readlines()
    return [line.strip() for line in data]


def vectorize(text, vocab):
    """Convert text to a vector based on the vocabulary."""
    vector = torch.zeros(len(vocab))
    for word in text.split():
        if word in vocab:
            vector[vocab[word]] += 1
    return vector


def main(args):
    # Load data from files
    source_train = load_data(args.source_train)
    target_train = load_data(args.target_train)
    source_dev = load_data(args.source_dev)
    target_dev = load_data(args.target_dev)
    source_test = load_data(args.source_test)
    target_test = load_data(args.target_test)

    # Basic tokenization and vectorization
    vocab = set(" ".join(source_train + source_dev + source_test).split())
    vocab = {word: i for i, word in enumerate(vocab)}

    # Vectorize the data
    X_train = torch.stack([vectorize(text, vocab) for text in source_train])
    y_train = torch.tensor(
        [0 if label == "negative" else 1 for label in target_train], dtype=torch.long
    )  # Assuming binary classification
    X_dev = torch.stack([vectorize(text, vocab) for text in source_dev])
    y_dev = torch.tensor(
        [0 if label == "negative" else 1 for label in target_dev], dtype=torch.long
    )
    X_test = torch.stack([vectorize(text, vocab) for text in source_test])
    y_test = torch.tensor(
        [0 if label == "negative" else 1 for label in target_test], dtype=torch.long
    )

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
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Validation loop
    with torch.no_grad():
        model.eval()
        dev_outputs = model(X_dev)
        dev_loss = criterion(dev_outputs, y_dev)
        print(f"Validation Loss: {dev_loss.item():.4f}")

    # Inference
    with torch.no_grad():
        model.eval()
        predictions = model(X_test)
        predicted_classes = torch.argmax(predictions, dim=1)
        print("Predicted classes:", predicted_classes)
        print("Actual classes:", y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_train",
        type=str,
        required=True,
        help="Path to the source training data.",
    )
    parser.add_argument(
        "--target_train",
        type=str,
        required=True,
        help="Path to the target training data.",
    )
    parser.add_argument(
        "--source_dev",
        type=str,
        required=True,
        help="Path to the source development data.",
    )
    parser.add_argument(
        "--target_dev",
        type=str,
        required=True,
        help="Path to the target development data.",
    )
    parser.add_argument(
        "--source_test", type=str, required=True, help="Path to the source test data."
    )
    parser.add_argument(
        "--target_test", type=str, required=True, help="Path to the target test data."
    )
    args = parser.parse_args()
    main(args)
