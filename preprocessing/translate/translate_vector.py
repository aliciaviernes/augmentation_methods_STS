import csv, datetime
from argostranslate import package, translate

from nltk.tokenize import word_tokenize

# Setting up translator
package.install_from_path('models/de_en.argosmodel')
installed_languages = translate.get_installed_languages()
translation_en_de = installed_languages[0].get_translation(installed_languages[1])
translation_de_en = installed_languages[1].get_translation(installed_languages[0])

vector = '../../data/datasets/vector_trainfile.csv'
langs = '../../data/tp_lookups/vector_langs.csv'


def translate_sentence(sentence, language):
    if language == 'de':
        en_sentence = translation_de_en.translate(sentence)
        return en_sentence
    elif language == 'en':
        de_sentence = translation_en_de.translate(sentence)
        return de_sentence
    else:
        return sentence

# FIRST
def read_save_csv(filename, delimiter):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        data = list(reader)
    return data


def write_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def append_to_right_list(lang, sentence, trans, en_list, de_list):
    if lang == 'de':
        en_list.append(trans)  # translation is English.
        de_list.append(sentence)  # original is German.
    elif lang == 'en':
        de_list.append(trans); en_list.append(sentence)
    else:
        de_list.append(sentence); en_list.append(sentence)
    return en_list, de_list

# THIRD
def write_separate_files(vectordata, langdata, outputfile_EN, outputfile_DE):
    for i in range(len(vectordata)):  # len(vectordata)
        l_sent = vectordata[i][1]; r_sent = vectordata[i][2]
        label = vectordata[i][3]
        l_lang = langdata[i][1]; r_lang = langdata[i][2]
        l_trans = translate_sentence(l_sent, l_lang)
        r_trans = translate_sentence(r_sent, r_lang)
        en_row = list(); de_row = list()
        en_row, de_row = append_to_right_list(l_lang, l_sent, l_trans, 
                                                en_row, de_row)
        en_row, de_row = append_to_right_list(r_lang, r_sent, r_trans, 
                                                en_row, de_row)
        de_row.append(label); en_row.append(label)
        if i % 100 == 0:
            print(f'{datetime.datetime.now()} {i} rows translated.')
        outputfile_EN.append(en_row); outputfile_DE.append(de_row)

# SECOND
def write_alignment_sample_vector(vectordata, langdata, filename):
    with open(filename, 'w') as f:
        for i in range(len(vectordata)):
            l_sent = vectordata[i][1]; r_sent = vectordata[i][2]
            l_lang = langdata[i][1]; r_lang = langdata[i][2]
            l_trans = translate_sentence(l_sent, l_lang)
            r_trans = translate_sentence(r_sent, r_lang)
            l_sent = ' '.join(word_tokenize(l_sent))
            r_sent = ' '.join(word_tokenize(r_sent))
            l_trans = ' '.join(word_tokenize(l_trans))
            r_trans = ' '.join(word_tokenize(r_trans))
            f.write(l_sent + ' ||| ' + l_trans + '\n')
            f.write(r_sent + ' ||| ' + r_trans + '\n')
            if i * 2 % 100 == 0:
                print(f'{datetime.datetime.now()} {i * 2} rows translated and appended.')


"""
    Executed for VECTOR file:
    - read_save_csv (vector_train, langs (important lookup))
    - cut header (vectordata = vectordata[1:])
    - write_alignment_sample_vector
    - additionally: write_separate_files (separating EN from DE)
"""
