import pandas as pd
import numpy as np
from PIL import Image
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_data = "data_256/"
path_metadata = "MAMe_metadata/"


def get_dict_labels():
    with open(path_metadata + 'MAMe_labels.csv', mode='r') as infile:
        reader = csv.reader(infile)
        dict_labels = dict((rows[1], int(rows[0])) for rows in reader)
    return dict_labels


def import_dataset():
    print('Import Dataset start')

    dataset_info = pd.read_csv(path_metadata + "MAMe_dataset.csv")

    dict_labels = get_dict_labels()

    x_train = []
    y_train = []
    x_validation = []
    y_validation = []

    print(f'Length: {len(dataset_info.index)}')

    for idx, row in dataset_info.iterrows():
        if idx % 1000 == 0:
            print(idx)

        im = np.asarray(Image.open(path_data + row['Image file']))
        y = dict_labels[row['Medium']]

        if row['Subset'] == 'train':
            x_train.append(im)
            y_train.append(y)

        if row['Subset'] == 'val':
            x_validation.append(im)
            y_validation.append(y)

    return x_train, y_train, x_validation, y_validation


def data_generator():
    """
    Create an image generator for the corresponding partition
    :param partition:
    :return:
    """
    print("Importing data generators")

    dataset_info = pd.read_csv(path_metadata + "MAMe_dataset.csv")

    dict_labels = get_dict_labels()
    train_df = pd.DataFrame(columns=["Image file", "Medium"])
    val_df = pd.DataFrame(columns=["Image file", "Medium"])

    print(f'Length: {len(dataset_info.index)}')

    for idx, row in dataset_info.iterrows():
        if idx % 1000 == 0:
            print(idx)

        y = dict_labels[row['Medium']]

        if row['Subset'] == 'train':
            train_df.append([row['Image file'], y])

        elif row['Subset'] == 'val':
            val_df.append([row['Image file'], y])

    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=path_data,
        batch_size=32,
        x_col="Image file",
        y_col="Medium",
        # seed=45
    )
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=path_data,
        batch_size=32,
        x_col="Image file",
        y_col="Medium",
        # seed=45
    )
    return train_generator, val_generator
