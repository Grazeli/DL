import csv
import pandas as pd
import os
from shutil import copyfile

path_data = "data_256/"
path_metadata = "MAMe_metadata/"
dataset_info = pd.read_csv(path_metadata + "MAMe_dataset.csv")

new_data_folder = 'data/'
sub_folders = ['train/', 'validation/', 'test/']

with open(path_metadata + 'MAMe_labels.csv', mode='r') as infile:
    reader = csv.reader(infile)
    dic_labels = dict((rows[1], int(rows[0])) for rows in reader)

# Create all the directories to separate images
if not os.path.isdir(new_data_folder):
    os.mkdir(new_data_folder)

for x in sub_folders:
    path_sub_folders = new_data_folder + x
    if not os.path.isdir(path_sub_folders):
        os.mkdir(path_sub_folders)

for key in dic_labels:
    for x in sub_folders:
        path_class_dir = new_data_folder + x + key

        if not os.path.isdir(path_class_dir):
            os.mkdir(path_class_dir)

# Move each image to its given directory
for idx, row in dataset_info.iterrows():
    y = row['Medium'] + '/'
    subset = row['Subset'] + '/'
    image_name = row['Image file']

    if subset == 'val/':
        subset = 'validation/'

    path_image = path_data + image_name
    new_path_folder = new_data_folder + subset + y

    if os.path.isfile(path_image) and os.path.isdir(new_path_folder):
        copyfile(path_image, new_path_folder + image_name)
