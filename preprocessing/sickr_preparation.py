import csv

txtpath = '../data/datasets/SICK/SICK_annotated.txt' # path to TXT file
csvpath = txtpath.replace('txt', 'csv')

with open(txtpath, 'r') as in_file: # path 
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
    with open(csvpath, 'w') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerows(lines)
