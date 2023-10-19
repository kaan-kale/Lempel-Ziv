from datasets import load_dataset
from utils import preprocess_text

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader


def load_split_of_dataset(dataset_name, split_name):
    """
    Load a specific split of a dataset using the Hugging Face datasets library.

    Args:
        dataset_name (str): The name of the dataset to load.
        split_name (str): The dataset split to load ("train," "test," or "validation").

    Returns:
        split_dataset: The specified split of the dataset.
    """
    dataset = load_dataset(dataset_name)
    split_dataset = dataset[split_name]
    return split_dataset


def preprocess_text_data(dataset, text_column="sentence", **kwargs):
    """
    Preprocess text data in a dataset using the preprocess_text function.

    Args:
        dataset: The dataset containing text data.
        text_column (str): The name of the column containing text to preprocess.

    Returns:
        preprocessed_dataset: The dataset with preprocessed text data.
    """
    # Preprocess the text data using the preprocess_text function
    normalize = kwargs.get("normalize", True)
    preprocessed_dataset = dataset.map(
        lambda example: {text_column: preprocess_text(example[text_column], normalize)}
    )
    return preprocessed_dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input = item["sentence"]
        label = item["label"]

        return input, label

    def collate_fn(self, batch):
        inputs = [torch.unsqueeze(torch.tensor(item[0]), -1) for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        return inputs, labels


def main():
    dataset = load_split_of_dataset("sst2", "validation")
    dataset = preprocess_text_data(dataset)
    dataset = CustomDataset(dataset)

    batch_size = 2
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    for batch in train_loader:
        # print(batch[0], batch[1])
        break

    return


if __name__ == "__main__":
    main()
