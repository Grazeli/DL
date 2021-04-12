import pandas as pd
import numpy as np
from PIL import Image
import csv

def importDataset():
    print('Import Dataset start')

    path_data = "data_256/"
    path_metadata = "MAMe_metadata/"
    dataset_info = pd.read_csv(path_metadata + "MAMe_dataset.csv")

    with open(path_metadata + 'MAMe_labels.csv', mode='r') as infile:
        reader = csv.reader(infile)
        dic_labels = dict((rows[1], int(rows[0])) for rows in reader)

    x_train = []
    y_train = []
    x_validation = []
    y_validation = []

    print(f'Length: {len(dataset_info.index)}')

    for idx, row in dataset_info.iterrows():
        if idx % 1000 == 0:
            print(idx)

        im = np.asarray(Image.open(path_data + row['Image file']))
        y = dic_labels[row['Medium']]

        if row['Subset'] == 'train':
            x_train.append(im)
            y_train.append(y)

        if row['Subset'] == 'val':
            x_validation.append(im)
            y_validation.append(y)

    return x_train, y_train, x_validation, y_validation