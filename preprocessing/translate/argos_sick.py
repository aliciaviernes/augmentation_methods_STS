from initial_imports import *

indices = {2, 4, 10, 11}

sickpath = path2datasets + 'SICK/SICK_annotated.txt'
sickdata = list() # 15 columns x 9841 rows

nr = 0; print(str(datetime.datetime.now()), '- Initialising...')

with open(sickpath, 'rt') as f:
    for line in f:
        x = line.split('\t')   # same.
        x[-1] = x[-1].rstrip()  # same.
        for index in indices:
            x[index] = translation_en_de.translate(x[index])
        sickdata.append(x); nr += 1  # TEST
        if nr % 500 == 0:
            dt = datetime.datetime.now()
            print(f'{dt} - {nr} rows processed.')

print(f'All {nr} rows are now translated.')  # same.
print('Writing new file...')  # same.

  # same.
with open(f'{path2datasets}SICK_annotated_DE.csv', 'w', newline='') as file:  # TEST
    writer = csv.writer(file, delimiter = '\t')
    for row in sickdata:
        writer.writerow(row)

print('File written.')
