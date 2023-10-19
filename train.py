import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import calculate_accuracy
from model import SimpleRNN
from dataset import preprocess_text_data, load_split_of_dataset, CustomDataset

from tqdm import tqdm

# Define a function to train the RNN model
def train_rnn(train_loader, model, optimizer, criterion):
    """Training function for the RNN model

    Args:
        train_loader :data loader for training data
        model : model to train
        optimizer : pytorch optimizer
        criterion : loss function

    Returns:
        loss: epoch loss
        accuracy: epoch accuracy
    """
    
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    for data, labels in tqdm(train_loader):
        optimizer.zero_grad()
        predictions = model(data)

        labels = (torch.reshape(labels, (-1, 1))).float()

        loss = criterion(predictions, labels)
        accuracy = calculate_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)


def main():
    # Load and preprocess your numeric data
    # Replace this with your own data and labels
    data = load_split_of_dataset("sst2", "train")
    data = preprocess_text_data(data, normalize=False)
    dataset = CustomDataset(data)
    batch_size = 16
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    # Initialize the RNN model
    input_dim = 1
    hidden_dim = 64
    output_dim = 1
    model = SimpleRNN(input_dim, hidden_dim, output_dim)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model for a few epochs
    N_EPOCHS = 5
    for epoch in range(N_EPOCHS):
        train_loss, train_accuracy = train_rnn(
            train_loader, model, optimizer, criterion
        )
        print(f"Epoch: {epoch+1:02}")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tTrain Accuracy: {train_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
