import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def load_txt_to_array(file_path):
    return np.loadtxt(file_path)

def process_txt_file(input_folder, txt_file):
    key = os.path.splitext(txt_file)[0]  # Remove the file extension to create the key
    file_path = os.path.join(input_folder, txt_file)
    matrix = load_txt_to_array(file_path).tolist()  # Convert the NumPy array to a Python list
    return key, matrix

def save_matrices_to_json(input_folder, output_file, num_processes=4):
    # Get a list of all txt files in the input folder
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    # Initialize an empty dictionary to store the matrices
    matrices = {}

    # Process each txt file using multiple processes
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_txt_file, input_folder, txt_file) for txt_file in txt_files]

        # Wait for all processes to complete and save the results to the dictionary
        for future in futures:
            key, matrix = future.result()
            matrices[key] = matrix

    # Save the dictionary to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(matrices, outfile)

if __name__ == "__main__":
    input_folder = './ids'  # Replace with your input folder
    output_file = 'ids.json'  # Replace with your desired output file name
    num_processes = 64
    save_matrices_to_json(input_folder, output_file, num_processes)