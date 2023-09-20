import os
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem

if __name__ == '__main__':
    path = r'/path/to/the/directory/of/this/project'

    calc = Calculator(descriptors, ignore_3D=False)
    SDFs = os.listdir(os.path.join(path, 'Data', 'SDFs'))
    mols_3d = [Chem.MolFromMolFile(os.path.join(path, 'Data', 'SDFs', '{}'.format(sdf))) for sdf in SDFs]
    result_3d = calc.map(mols_3d)
    result_3d = list(result_3d)
    data = [result_3d[i].fill_missing(0).asdict() for i in range(len(result_3d))]
    df = pd.DataFrame(data, index=[SDFs[i].strip('.sdf') for i in range(len(SDFs))])
    df.to_csv(os.path.join(path, 'MolDescriptors_3D.tsv'), sep='\t')