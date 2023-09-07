import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import autograd


def create_custom_dataloader(data_path, descriptors_path, batch_size, device):
    """
    Create a custom DataLoader for data and conditions.

    Args:
        data_path (str): Path to the data file.
        descriptors_path (str): Path to the molecular descriptors file.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data during training.

    Returns:
        DataLoader: A DataLoader instance for your data and conditions.
    """
    # Read data from files
    data = pd.read_csv(data_path, sep="\t")
    descriptors = pd.read_csv(descriptors_path, index_col=0, sep="\t")

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    descriptors = pd.DataFrame(scaler.fit_transform(descriptors), columns=descriptors.columns, index=descriptors.index)
    data = data.iloc[:, 0:3].join(pd.DataFrame(scaler.fit_transform(data.iloc[:, 3:]), columns=data.columns[3:]))

    S = pd.DataFrame(columns=descriptors.columns)
    M = pd.DataFrame(columns=data.columns[3:])
    T = []
    D = []
    for i in range(len(data)):
        if data.iloc[i].COMPOUND_NAME in descriptors.index:
            S = pd.concat([S, descriptors[descriptors.index == data.iloc[i].COMPOUND_NAME]])
            subset_data = data.iloc[i, 3:].to_frame().T
            M = pd.concat([M, subset_data], ignore_index=True)
            T.append(Time(data.iloc[i].SACRI_PERIOD))
            D.append(Dose(data.iloc[i].DOSE_LEVEL))

    # Convert data to PyTorch tensors
    S = torch.tensor(S.to_numpy(dtype=np.float32), device=device)
    M = torch.tensor(M.to_numpy(dtype=np.float32), device=device)
    T = scaler.fit_transform(np.array(T, dtype=np.float32).reshape(len(T), -1))
    T = torch.tensor(T, device=device)
    D = scaler.fit_transform(np.array(D, dtype=np.float32).reshape(len(D), -1))
    D = torch.tensor(D, device=device)
    dataset = torch.utils.data.TensorDataset(M, S, T, D)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def Time(SACRIFICE_PERIOD):
    switcher = {
        '4 day': 4 / 29,
        '8 day': 8 / 29,
        '15 day': 15 / 29,
        '29 day': 29 / 29
    }
    return switcher.get(SACRIFICE_PERIOD, 'error')

def Dose(DOSE_LEVEL):
    switcher = {
        'Low': 0.1,
        'Middle': 0.3,
        'High': 1
    }
    return switcher.get(DOSE_LEVEL, 'error')
#
# def load_generator(input_dim, output_dim, num_classes, filename, device):
#     # Loading a pretrained generator
#
# def generate_samples(generator, num_samples, target_class, device):
#     # Generate samples for a specific class
#
# def save_generated_records(generated_samples, filename):
#     # Save generated images
