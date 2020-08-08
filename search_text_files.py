import re
import pandas as pd
import os
from nltk.corpus import stopwords
from collections import Counter
import string
from string import printable

def return_date(data_location):
    data = open(data_location, encoding='utf-8', errors='ignore').read()
    lines_of_data = data.splitlines()
    container = []
    for line in lines_of_data:
        pubdate = ''
        line = line.strip()
        if 'IL.pubdate:' in line:
            pubdate += line[12:22]
            container.append(pubdate)
            continue
    return container


def return_body(data_location):
    data = open(data_location, encoding='utf-8', errors='ignore').read()
    lines_of_data = data.splitlines()
    active = False
    stored = []
    counter = 0
    for line in lines_of_data:
        line = line.strip()
        line = re.sub('‘’""–', '', line)
        counter += 1
        if line.startswith("Document:"):
            tmp = ''
            active = True    # now we want to do things
            continue         # but not in this loop
        if line.startswith("===EOD==="):
            stored.append(tmp)
            active = False
            continue
        if active:
            tmp += line
    return stored


def return_source(data_location):
    data = open(data_location, encoding='utf-8', errors='ignore').read()
    lines_of_data = data.splitlines()
    active = False
    stored = []
    for line in lines_of_data:
        line = line.strip()
        line = re.sub('[‘’""–]', '', line)
        if line.startswith("feed:"):
            tmp = ''
            active = True    # now we want to do things
            continue         # but not in this loop
        if line.startswith("date:"):
            stored.append(tmp)
            active = False
            continue
        if active:
            tmp += line[8::]
    return stored


def return_title_tb(data_location):
    data = open(data_location, encoding='utf-8', errors='ignore').read()
    lines_of_data = data.splitlines()
    active = False
    stored = []
    for line in lines_of_data:
        line = line.strip()
        line = re.sub('[‘’""–]', '', line)
        if line.startswith("dateposted:"):
            tmp = ''
            active = True    # now we want to do things
            continue         # but not in this loop
        if line.startswith("Document:"):
            stored.append(tmp)
            active = False
            continue
        if active:
            tmp += line
    return stored


def return_title_osc(data_location):
    data = open(data_location, encoding='utf-8', errors='ignore').read()
    lines_of_data = data.splitlines()
    active = False
    stored = []
    for line in lines_of_data:
        line = line.strip()
        line = re.sub('[‘’""–]', '', line)
        if line.startswith("hastext:"):
            tmp = ''
            active = True    # now we want to do things
            continue         # but not in this loop
        if line.startswith("IL.docid:"):
            stored.append(tmp)
            active = False
            continue
        if active:
            tmp += line
    return stored


# > 2018-07-03
def return_tb_title_t2(data_location):
    data = open(data_location, encoding='utf-8', errors='ignore').read()
    lines_of_data = data.splitlines()
    active = False
    stored = []
    for line in lines_of_data:
        line = line.strip()
        line = re.sub('[‘’""–]', '', line)
        if line.startswith("The body of this product is"):
            tmp = ''
            active = True    # now we want to do things
            continue         # but not in this loop
        if line.startswith("BODY"):
            stored.append(tmp)
            active = False
            continue
        if active:
            tmp += line
    return stored


def return_body_t2(data_location):
    data = open(data_location, encoding='utf-8', errors='ignore').read()
    lines_of_data = data.splitlines()
    active = False
    stored = []
    for line in lines_of_data:
        line = line.strip()
        line = re.sub('[‘’""–]', '', line)
        if line.startswith("BODY"):
            tmp = ''
            active = True    # now we want to do things
            continue         # but not in this loop
        if line.startswith("SOURCE DESCRIPTOR"):
            stored.append(tmp)
            active = False
            continue
        if active:
            tmp += line
    return stored


def return_source_t2(data_location):
    data = open(data_location, encoding='utf-8', errors='ignore').read()
    lines_of_data = data.splitlines()
    active = False
    stored = []
    for line in lines_of_data:
        line = line.strip()
        line = re.sub('[‘’""–]', '', line)
        if line.startswith("SOURCE DESCRIPTOR"):
            tmp = ''
            active = True    # now we want to do things
            continue         # but not in this loop
        if line.startswith("===EOD==="):
            stored.append(tmp)
            active = False
            continue
        if active:
            tmp += line
    return stored


# build df
file_names = [f for f in os.listdir('C:/Users/Andrew/Desktop/nlp_corpus_stack/')]

# shell df to append into
df = pd.DataFrame({'date': [], 'source': [],
                   'body': [], 'title_tb': [],
                   'title_osc': [], 'filename': []})

# loop through text files
for i in file_names:
    check_date = return_date('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
    if check_date[0] <= '2018-07-03':

        dates = return_date('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        source = return_source('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        body = return_body('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        title_osc = return_title_osc('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        title_tb = return_title_tb('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        file_name_rep = [i for x in range(len(dates))]
        data = pd.DataFrame({'date': dates, 'title_osc': title_osc, 'source': source,
                             'body': body, 'title_tb': title_tb,
                             'filename': file_name_rep})
        df = df.append(data)
    else:
        dates = return_date('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        source = return_source_t2('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        body = return_body_t2('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        title_osc = return_title_osc('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        title_tb = return_tb_title_t2('C:/Users/Andrew/Desktop/nlp_corpus_stack/' + i)
        file_name_rep = [i for x in range(len(dates))]
        data = pd.DataFrame({'date': dates, 'title_osc': title_osc, 'source': source,
                             'body': body, 'title_tb': title_tb,
                             'filename': file_name_rep})
        df = df.append(data)
