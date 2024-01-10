import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def generate(treatments, descriptors, training_data, descriptors_training, result_path, generator, opt, device):
    '''
    Args:
    treatments (pd.DataFrame): treatment conditions of interest
    discriptors (pd.DataFrame): molecular descriptors of the compounds of interest
    training_data (pd.DataFrame): all the training data used for the pretrained model
    descriptors_training (pd.DataFrame): molecular descriptors of the compounds used to train the model
    result_path (str) : path to the file where you want to store the results
    '''

    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(descriptors_training)
    scaled_MDs = pd.DataFrame(scaler.transform(descriptors), columns=descriptors.columns, index=descriptors.index)
    S = pd.DataFrame()
    for i in range(len(treatments)):
        S = pd.concat([S, scaled_MDs[scaled_MDs.index == treatments.iloc[i].COMPOUND_NAME]])
    S = torch.tensor(S.to_numpy(dtype=np.float32), device=device)

    scaler.fit(training_data['SACRI_PERIOD'].apply(Time).to_numpy(dtype=np.float32).reshape(-1, 1))
    T = scaler.transform(treatments['SACRI_PERIOD'].apply(Time).to_numpy(dtype=np.float32).reshape(-1, 1))
    T = torch.tensor(T, device=device)

    scaler.fit(training_data['DOSE_LEVEL'].apply(Dose).to_numpy(dtype=np.float32).reshape(-1, 1))
    D = scaler.transform(treatments['DOSE_LEVEL'].apply(Dose).to_numpy(dtype=np.float32).reshape(-1, 1))
    D = torch.tensor(D, device=device)

    measurements = training_data.iloc[:, 3:]
    scaler.fit(measurements)
    Results = pd.DataFrame(columns=measurements.columns)
    for i in range(S.shape[0]):
        num = 0
        while num < opt.num_generate:
            z = torch.randn(1, opt.Z_dim).to(device)
            generated_records = generator(z, S[i].view(1, -1), T[i].view(1, -1), D[i].view(1, -1))
            generated_records = scaler.inverse_transform(generated_records.cpu().detach().numpy())
            check = np.sum(generated_records[:, 9:14])
            if 95 < check < 105:
                num += 1
                Results.loc[i] = generated_records.flatten()
    Results = pd.concat([treatments.loc[treatments.index.repeat(opt.num_generate)].reset_index(drop=True), Results], axis=1)
    Results.to_csv(result_path, sep='\t', index=False)


if __name__ == '__main__':
    import os
    from opt import parse_opt
    from model import Generator
    from utils import Time, Dose
    import pandas as pd

    path = './'
    opt = parse_opt()
    num = opt.num_generate
    Z_dim = opt.Z_dim
    Stru_dim = opt.Stru_dim
    Time_dim = opt.Time_dim
    Dose_dim = opt.Dose_dim
    Measurement_dim = opt.Measurement_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    treatments = pd.read_csv(os.path.join(path, 'Data', 'Example_Treatments_test.tsv'), sep="\t")
    training_data = pd.read_csv(os.path.join(path, 'Data', 'Example_Data_training.tsv'), sep="\t")  ### this file should store all the training data used for the pretrained model
    MDs = pd.read_csv(os.path.join(path, 'Data', 'Example_MolecularDescriptors.tsv'), index_col=0, sep="\t")
    descriptors_training = MDs[MDs.index.isin(training_data['COMPOUND_NAME'])]
    descriptors = MDs[MDs.index.isin(treatments['COMPOUND_NAME'])]

    generator = Generator(Z_dim, Stru_dim, Time_dim, Dose_dim, Measurement_dim).to(device)

    model_path = r'./models'  # please change it to the folder where your well-trained model save
    weights = torch.load(os.path.join(model_path, 'AnimalGAN'))
    generator.load_state_dict(weights)
    generator.eval()

    result_path = os.path.join(path, 'Results', 'generated_data_{}.tsv'.format(opt.num_generate))
    generate(treatments, descriptors, training_data, descriptors_training, result_path, generator, opt, device)
