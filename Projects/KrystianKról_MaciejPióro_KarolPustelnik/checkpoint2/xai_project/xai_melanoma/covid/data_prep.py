import os
import pandas as pd

# constants:


dataset_dir = '/home/kpusteln/covid_dataset'



test_filenames = []
train_filenames = []
val_filenames = []

for directory in os.listdir(dataset_dir):
    if directory == 'Test':
        for subdirectory in os.listdir(dataset_dir + '/' + directory):
            if subdirectory == 'COVID-19':
                for file in os.listdir(dataset_dir + '/' + directory + '/' + subdirectory + '/images'):
                    filename = os.fsdecode(file)
                    pair = (filename, 0)
                    test_filenames.append(pair)
            elif subdirectory == 'Non-COVID':
                for file in os.listdir(dataset_dir + '/' + directory + '/' + subdirectory + '/images'):
                    filename = os.fsdecode(file)
                    pair = (filename, 1)
                    test_filenames.append(pair)
            elif subdirectory == 'Normal':
                for file in os.listdir(dataset_dir + '/' + directory + '/' + subdirectory + '/images'):
                    filename = os.fsdecode(file)
                    pair = (filename, 2)
                    test_filenames.append(pair)
    elif directory == 'Train':
        for subdirectory in os.listdir(dataset_dir + '/' + directory):
            if subdirectory == 'COVID-19':
                for file in os.listdir(dataset_dir + '/' + directory + '/' + subdirectory + '/images'):
                    filename = os.fsdecode(file)
                    pair = (filename, 0)
                    train_filenames.append(pair)
            elif subdirectory == 'Non-COVID':
                for file in os.listdir(dataset_dir + '/' + directory + '/' + subdirectory + '/images'):
                    filename = os.fsdecode(file)
                    pair = (filename, 1)
                    train_filenames.append(pair)
            elif subdirectory == 'Normal':
                for file in os.listdir(dataset_dir + '/' + directory + '/' + subdirectory + '/images'):
                    filename = os.fsdecode(file)
                    pair = (filename, 2)
                    train_filenames.append(pair)
    elif directory == 'Val':
        for subdirectory in os.listdir(dataset_dir + '/' + directory):
            if subdirectory == 'COVID-19':
                for file in os.listdir(dataset_dir + '/' + directory + '/' + subdirectory + '/images'):
                    filename = os.fsdecode(file)
                    pair = (filename, 0)
                    val_filenames.append(pair)
            elif subdirectory == 'Non-COVID':
                for file in os.listdir(dataset_dir + '/' + directory + '/' + subdirectory + '/images'):
                    filename = os.fsdecode(file)
                    pair = (filename, 1)
                    val_filenames.append(pair)
            elif subdirectory == 'Normal':
                for file in os.listdir(dataset_dir + '/' + directory + '/' + subdirectory + '/images'):
                    filename = os.fsdecode(file)
                    pair = (filename, 2)
                    val_filenames.append(pair)
                    
                    
                    
test_df = pd.DataFrame(test_filenames, columns=['filename', 'label'])
train_df = pd.DataFrame(train_filenames, columns=['filename', 'label'])
val_df = pd.DataFrame(val_filenames, columns=['filename', 'label'])

test_df.to_csv('/home/kpusteln/covid_dataset/test.csv', index=False)
train_df.to_csv('/home/kpusteln/covid_dataset/train.csv', index=False)
val_df.to_csv('/home/kpusteln/covid_dataset/val.csv', index=False)
