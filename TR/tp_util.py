from baseline.util import *
import time, random

from nltk.tokenize import word_tokenize

starttime = time.time()

################################# ALIGNMENT TIME #############################################

def write_alignment_sample(train_SRC, train_TGT, outputname):  # prepare file for awesome-align
    f = open(outputname, 'w')
    for i in range(len(train_SRC)):
        left_source = ' '.join(word_tokenize(train_SRC[i].texts[0])) 
        left_target = ' '.join(word_tokenize(train_TGT[i].texts[0]))
        right_source = ' '.join(word_tokenize(train_SRC[i].texts[1]))
        right_target = ' '.join(word_tokenize(train_TGT[i].texts[1]))
        f.write(left_source + ' ||| ' + left_target + '\n')
        f.write(right_source + ' ||| ' + right_target + '\n')
    f.close()


def alignment_str2list(line):  # reads every line
    src_positions, tgt_positions = list(), list()
    line = line.rstrip().split(' ')
    for pair in line:
        pair = pair.split('-')
        src_positions.append(int(pair[0])); tgt_positions.append(int(pair[1]))
    return src_positions, tgt_positions


def alignment_lists(alignmentfile):  # creates two lists of lists
    SRC_positions, TGT_positions = list(), list()
    with open(alignmentfile, 'r') as f:
        for line in f:
            src_positions, tgt_positions = alignment_str2list(line)
            SRC_positions.append(src_positions) 
            TGT_positions.append(tgt_positions)
    return SRC_positions, TGT_positions


def alignment_dict(src_positionlist, tgt_positionlist):  # create dict as we want it
    al_dict = dict()
    for i in range(len(src_positionlist)):
        if src_positionlist[i] not in al_dict:
            al_dict[src_positionlist[i]] = tgt_positionlist[i]
        else:
            if type(al_dict[src_positionlist[i]]) == set:
                al_dict[src_positionlist[i]].add(tgt_positionlist[i])
            else:
                inset = set()
                inset.add(al_dict[src_positionlist[i]])
                inset.add(tgt_positionlist[i])
                al_dict[src_positionlist[i]] = inset
    return al_dict

################################### READ TEXT ###############################################

def sentences2list(line):  # splits and tokenizes every line
    line = line.rstrip().split('|||')
    return word_tokenize(line[0]), word_tokenize(line[1])


def sentence_lists(sentencefile):  # creates two lists of lists
    SRC_sentences, TGT_sentences = list(), list()
    with open(sentencefile, 'r') as f:
        for line in f:
            src_sent, tgt_sent = sentences2list(line)
            SRC_sentences.append(src_sent)
            TGT_sentences.append(tgt_sent)
    return SRC_sentences, TGT_sentences

################################### MULTI ALIGNMENT #########################################

def multialignment_target(al_dict, s):  # NEW
    t = al_dict[s]
    return sorted(list(t))


def multialignment_source(al_dict, p):  # NEW
    t = al_dict[p]  # find target position
    src_pos = [k for k,v in al_dict.items() if v == t]  # check if target comes from multiple sources.
    src_pos = src_pos[0] if len(src_pos) == 1 else sorted(src_pos)  # return int if one position, else list
    return src_pos, t


def positions2string(poslist, tokenlist):  # we don't want that though.
    string = ''
    for p in poslist:
        string += tokenlist[p] + ' '
    return string.rstrip()

################################ MAIN FUNCTION HELPERS ######################################

def position_alignment(src_pos, tgt_pos, src_sent, tgt_sent, p): # this is only for one sentence and one word
    idx_S = src_pos.index(p)
    idx_T = tgt_pos[idx_S]
    return tgt_sent[idx_T] 


def randomly_choose_positions(src_pos, a=0.2):
    positions = set(); candidates = list()
    for i in src_pos:
        if i != max(src_pos):
            candidates.append(i)
    num_tp = max(int(len(src_pos) * a), 1)
    for _ in range(num_tp):
        positions.add(random.choice(candidates))
    return positions


def source_target_comparison(src_sent, tgt_sent):  # deprecated
    dif_s_t = set(src_sent) - set(tgt_sent)
    dif_t_s = set(tgt_sent) - set(src_sent)
    return len(dif_s_t), len(dif_t_s)

############################### ACTUAL ALIGNMENTS ###########################################

def trans_place(src_sent, tgt_sent, src_pos, tgt_pos, c=0):
    al_dict = alignment_dict(src_pos, tgt_pos)
    pos = randomly_choose_positions(src_pos, a=0.2)  # choose randomly
    tbm = src_sent.copy()
    for p in pos:
        if type(al_dict[p]) == set:
            t = multialignment_target(al_dict, p)
            str_t = positions2string(t, tgt_sent)  # this is a bit problematic
            tbm[p] = str_t
        else:
            s, t = multialignment_source(al_dict, p)
            if type(s) == int:
                tbm[s] = tgt_sent[t]
            else:
                tbm[s[0]] = tgt_sent[t]
                for i in s[1:]:
                    tbm[i] = ''
    if '' in tbm:
        tbm.remove('')
    # NEW ADDITION START
    if tbm == src_sent:
        if c <= 3:
            c +=1
            tbm = trans_place(src_sent, tgt_sent, src_pos, tgt_pos, c=c)
    # NEW ADDITION END
    return tbm


# Main function - missing: nr of augmentations
def trans_place_all(alignmentfile, sentencefile):  # NOTE show what kind of argument this is 
    augmentations = list()
    SRC_positions, TGT_positions = alignment_lists(alignmentfile)  # lists of positions
    SRC_sentences, TGT_sentences = sentence_lists(sentencefile)  # lists of alignments
    for i in range(len(SRC_positions)):  # for each sentence...
        new_sent = trans_place(src_sent=SRC_sentences[i], tgt_sent=TGT_sentences[i],
                                src_pos=SRC_positions[i], tgt_pos=TGT_positions[i])
        augmentations.append(' '.join(new_sent))
    return augmentations
