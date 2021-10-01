from langdetect import detect_langs
import pandas as pd
import time, csv
# import util

starttime = time.time()

path2datasets = '/path/to/datasets/'
vector = f'{path2datasets}vector/vector_trainfile.csv'

traindata = pd.read_csv(vector)
relevant_headers = {'textA', 'textB'}

detected_languages = list()


def lang_formatting(langlist):
    langdict = dict()
    for lang in langlist:
        lang = str(lang).split(':')
        langdict[lang[0]] = round(float(lang[1]), 3)
    return langdict

# a function from the vaults :')
def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]


def pickout_langs(langdict):
    targets = {'de', 'en'}
    modified_langdict = dict()
    for lang in langdict:
        if lang in targets:
            modified_langdict[lang] = langdict[lang]
    if modified_langdict != dict():
        return keywithmaxval(modified_langdict)
    else:
        return 'check'
        

for i in range(len(traindata)):  # len(traindata)
    # print(i)
    textA = traindata.iloc[i]['textA']
    textB = traindata.iloc[i]['textB']
    langA = pickout_langs(lang_formatting(detect_langs(textA)))
    langB = pickout_langs(lang_formatting(detect_langs(textB)))
    detected_languages.append([i, langA, langB])


with open('vector_langs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['index', 'lang_textA', 'lang_textB'])
    for row in detected_languages:
        spamwriter.writerow(row)
