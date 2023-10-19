# import zlib
import torch

# def compress_data(data):
#     compressed_data = zlib.compress(data.encode(), level=zlib.Z_BEST_COMPRESSION)
#     return compressed_data

# def decompress_data(compressed_data):
#     decompressed_data = zlib.decompress(compressed_data)
#     return decompressed_data.decode()


from compression import compress

def preprocess_text(text, normalize=True):
    compressed_text = compress(text)
    n = 255 if normalize else 1
    preprocessed_text = [byte/n for byte in compressed_text]
    return preprocessed_text


# Define a function to calculate accuracy
def calculate_accuracy(predictions, labels):
    # Round predictions to the nearest integer (0 or 1)
    rounded_predictions = torch.squeeze(torch.round(predictions), 1)
    labels = torch.squeeze(labels, -1)
    correct = (rounded_predictions.eq(labels)).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.cpu()