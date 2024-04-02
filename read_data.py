import pandas as pd
import os
import numpy as np
import glob

def read_data(data_dir='data'):

    data_dict = {}
    lst_dir = sorted(glob.glob('data_Test/*'),key=os.path.getmtime)
    # Loop through each subdirectory in data_dir
    for sub_dir_path in lst_dir:
        # sub_dir_path = os.path.join(data_dir, sub_dir)
        sub_dir = os.path.basename(sub_dir_path)
        if os.path.isdir(sub_dir_path):
            # Initialize list to store values from pickle files
            values = []
            # Loop through each pickle file in the subdirectory
            for file in os.listdir(sub_dir_path):
                file_path = os.path.join(sub_dir_path, file)
                if file.endswith(".pkl") and os.path.isfile(file_path):
                    # Read the values from the pickle file and append to list
                    values.append(pd.read_pickle(file_path))
            # Add the list of values to the dictionary with the subdirectory name as key

            values = np.array(values)
            data_dict[sub_dir] = values.mean(axis=0)

    return data_dict

if __name__ == '__main__':
    data = read_data()
    print(data.keys())