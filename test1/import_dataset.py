import pandas as pd
import numpy as np
from PIL import Image
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_data = "../data_256/"
path_metadata = "../MAMe_metadata/"


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


def create_dataframes_csv():
    """
    Create all dataframes and export them to csv files to load them easier and faster later on
    """
    print("Importing data generators")

    dataset_info = pd.read_csv(path_metadata + "MAMe_dataset.csv")

    train_df = pd.DataFrame(columns=["Image file", "Medium"])
    val_df = pd.DataFrame(columns=["Image file", "Medium"])
    test_df = pd.DataFrame(columns=["Image file", "Medium"])

    print(f'Length: {len(dataset_info.index)}')

    for idx, row in dataset_info.iterrows():
        if idx % 1000 == 0:
            print(idx)

        y = row['Medium'].strip()

        if row['Subset'] == 'train':
            train_df = train_df.append({"Image file": row['Image file'], "Medium": y}, ignore_index=True)

        elif row['Subset'] == 'val':
            val_df = val_df.append({"Image file": row['Image file'], "Medium": y}, ignore_index=True)

        elif row['Subset'] == 'test':
            test_df = test_df.append({"Image file": row['Image file'], "Medium": y}, ignore_index=True)


    train_df.to_csv(f"{path_metadata}/train_df.csv", index=False)
    val_df.to_csv(f"{path_metadata}/val_df.csv", index=False)
    test_df.to_csv(f"{path_metadata}/test_df.csv", index=False)


def get_validation_from_csv():
    val_df = pd.read_csv(f"{path_metadata}/val_df.csv")
    val_df = pd.get_dummies(val_df, columns=["Medium"], prefix="", prefix_sep="")
    x_validation = [np.asarray(Image.open(f"{path_data}/{image_name}")) / 255 for image_name in val_df["Image file"]]
    y_validation = val_df.values[:, 1:]
    return np.array(x_validation), np.array(y_validation)


def preprocess_image(image):
    return image/255


def data_generators(batch_size):
    train_df = pd.read_csv(f"{path_metadata}/train_df.csv")
    val_df = pd.read_csv(f"{path_metadata}/val_df.csv")
    # columns = list(get_dict_labels().keys())
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
    print("Creating train generator")
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=path_data,
        batch_size=batch_size,
        x_col="Image file",
        y_col="Medium",
        class_mode="categorical",
        shuffle=True
        # seed=45
    )
    print("Creating val generator")
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=path_data,
        batch_size=batch_size,
        x_col="Image file",
        y_col="Medium",
        class_mode="categorical",
        shuffle=False,
        # seed=45
    )
    return train_generator, val_generator
