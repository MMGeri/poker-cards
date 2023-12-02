import os
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import csv

def generate_dct_filter(size, i, j):
    filter_matrix = np.zeros((size, size))

    for x in range(size):
        for y in range(size):
            filter_matrix[x, y] = np.cos((2 * x + 1) * i * np.pi / (2 * size)) * np.cos((2 * y + 1) * j * np.pi / (2 * size))

    return filter_matrix

def plot_dct_filter(filter_matrix, title, save_path=None):
    plt.imshow(filter_matrix, cmap='gray', interpolation='nearest')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def create_csv(output_directory):
    csv_file_path = os.path.join(output_directory, 'dct_filtersdct_filters.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write data
        for i in range(filter_size):
            for j in range(filter_size):
                filename = f'dct_filter_{i}_{j}.png'
                label = 'nope'
                writer.writerow({'filename': filename, 'label': label})

    print(f'CSV file created at: {csv_file_path}')

if __name__ == "__main__":
    filter_size = 16

    # Create a directory to store the filter images and CSV file
    output_directory = '../training/data'
    os.makedirs(output_directory+'/dct_filters', exist_ok=True)

    for i in range(filter_size):
        print('working..')
        for j in range(filter_size):
            # Generate DCT filter
            dct_filter = generate_dct_filter(filter_size, i, j)

            # Plot and save DCT filter
            save_path = os.path.join(output_directory+'/dct_filters', f'dct_filter_{i}_{j}.png')
            plot_dct_filter(dct_filter, f'DCT Filter {i}_{j}', save_path)

    # Create CSV file
    create_csv('../training/data/')
