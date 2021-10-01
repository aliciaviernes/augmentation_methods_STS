from initial_imports import *
import csv

data = list() 
nr = 0
parts = ['train', 'test']
indices = {-1, -2}

for part in parts:

    print(str(datetime.datetime.now()), f'- Initialising {part}...')

    with open(path2datasets + f'/MSRP/msr_paraphrase_{part}.txt', 'rt') as f:  # TEST
        for line in f:
            x = line.split('\t')  # same.
            x[-1] = x[-1].rstrip()  # random empty line  # same.
            for index in indices:
                x[index] = translation_en_de.translate(x[index])
            # x[-1] = translation_en_de.translate(x[-1])
            # x[-2] = translation_en_de.translate(x[-2])
            data.append(x); nr += 1  # TEST  # same.
            if nr % 500 == 0:  # same.
                dt = datetime.datetime.now()  # same.
                print(f'{dt} - {nr} rows processed.')  # same.

    print(f'All {nr} rows are now translated.')  # same.
    print('Writing new file...')  # same.

    # same.
    with open(f'{path2datasets}/MSRP/msr_paraphrase_{part}_DE.csv', 'w', newline='') as file:  # TEST
        writer = csv.writer(file, delimiter = '\t')
        for row in data:
            writer.writerow(row)

    # same.
    print('File written.')
