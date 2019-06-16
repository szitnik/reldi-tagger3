#!/usr/bin/python
#-*-coding:utf8-*-

"""
Usage as library:

    reldiTagger = ReldiTagger3()
    reldiTagger.load_models("sl", lemmatise=True)
    reldiTagger.processSentence([['1.2.1.25-29', 'Kupil'], ['1.2.2.31-33', 'sem'], ['1.2.3.35-40', 'banane'], ['1.2.4.42-43', 'in'], ['1.2.5.45-48', 'kruh'], ['1.2.6.49-49', '.']])

Parameter for the processSentence can be retrieved by the reldi-tokeniser3.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os


from train_tagger import extract_features_msd
from train_lemmatiser import extract_features_lemma
from subprocess import Popen, PIPE
import pickle
import pycrfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class ReldiTagger3:

    def __init__(self):
        self.trie = None
        self.tagger = None
        self.lemmatiser = None
        self.reldir = os.path.dirname(os.path.abspath(__file__))

    def load_models(self, lang, lemmatise=True, dir=None):
        if dir != None:
            self.reldir = dir
        self.trie = pickle.load(open(os.path.join(self.reldir, lang + '.marisa'), 'rb'), fix_imports=True, encoding="bytes")
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(os.path.join(self.reldir, lang + '.msd.model'))
        if lemmatise:
            self.lemmatiser = {'model': pickle.load(open(os.path.join(self.reldir, lang + '.lexicon.guesser'), 'rb'), fix_imports=True,
                                     encoding="bytes"),
                      'lexicon': pickle.load(open(os.path.join(self.reldir, lang + '.lexicon'), 'rb'), fix_imports=True,
                                       encoding="bytes")}
        else:
            self.lemmatiser = None

    def tag_sent(self, sent):
        return self.tagger.tag(extract_features_msd(sent, self.trie))


    def tag_lemmatise_sent(self, sent):
        return [(a, self.get_lemma(b, a)) for a, b in zip(self.tag_sent(sent), sent)]


    def get_lemma(self, token, msd):
        lexicon = self.lemmatiser['lexicon']
        key = token.lower() + '_' + msd
        if key in lexicon:
            return lexicon[key][0].decode('utf8')
        if msd[:2] != 'Np':
            for i in range(len(msd) - 1):
                for key in lexicon.keys(key[:-(i + 1)]):
                    return lexicon[key][0].decode('utf8')
        return self.guess_lemma(token, msd)


    def guess_lemma(self, token, msd):
        if len(token) < 3:
            return self.apply_rule(token, "(0,'',0,'')", msd)
        model = self.lemmatiser['model']
        if msd not in model:
            return token
        else:
            lemma = self.apply_rule(token, model[msd].predict(
                extract_features_lemma(token))[0], msd)
            if len(lemma) > 0:
                return lemma
            else:
                return token


    def suffix(self, token, n):
        if len(token) > n:
            return token[-n:]


    def apply_rule(self, token, rule, msd):
        rule = list(eval(rule))
        if msd:
            if msd[:2] == 'Np':
                lemma = token
            else:
                lemma = token.lower()
        else:
            lemma = token.lower()
        rule[2] = len(token) - rule[2]
        lemma = rule[1] + lemma[rule[0]:rule[2]] + rule[3]
        return lemma

    def processSentence(self, entryList):
        totag = []
        for token in [e[-1] for e in entryList]:
            if ' ' in token:
                if len(token) > 1:
                    totag.extend(token.split(' '))
            else:
                totag.append(token)
        tag_counter = 0
        if self.lemmatiser == None:
            tags = self.tag_sent(totag)
            tags_proper = []
            for token in [e[-1] for e in entryList]:
                if ' ' in token:
                    if len(token) == 1:
                        tags_proper.append(' ')
                    else:
                        tags_proper.append(
                            ' '.join(tags[tag_counter:tag_counter + token.count(' ') + 1]))
                        tag_counter += token.count(' ') + 1
                else:
                    tags_proper.append(tags[tag_counter])
                    tag_counter += 1
            sys.stdout.write(u''.join(['\t'.join(
                entry) + '\t' + tag + '\n' for entry, tag in zip(entryList, tags_proper)]) + '\n')
        else:
            tags = self.tag_lemmatise_sent(totag)
            tags_proper = []
            for token in [e[-1] for e in entryList]:
                if ' ' in token:
                    if len(token) == 1:
                        tags_proper.append([' ', ' '])
                    else:
                        tags_temp = tags[
                            tag_counter:tag_counter + token.count(' ') + 1]
                        tag = ' '.join([e[0] for e in tags_temp])
                        lemma = ' '.join([e[1] for e in tags_temp])
                        tags_proper.append([tag, lemma])
                        tag_counter += token.count(' ') + 1
                else:
                    tags_proper.append(tags[tag_counter])
                    tag_counter += 1
            sys.stdout.write(''.join(['\t'.join(entry) + '\t' + tag[0] + '\t' + tag[
                          1] + '\n' for entry, tag in zip(entryList, tags_proper)]) + '\n')

    def read_and_write(self, istream, index, ostream):
        entry_list = []
        sents = []
        for line in istream:
            if line.strip() == '':
                totag = []
                for token in [e[index] for e in entry_list]:
                    if ' ' in token:
                        if len(token) > 1:
                            totag.extend(token.split(' '))
                    else:
                        totag.append(token)
                tag_counter = 0
                if self.lemmatiser == None:
                    tags = self.tag_sent(totag)
                    tags_proper = []
                    for token in [e[index] for e in entry_list]:
                        if ' ' in token:
                            if len(token) == 1:
                                tags_proper.append(' ')
                            else:
                                tags_proper.append(
                                    ' '.join(tags[tag_counter:tag_counter + token.count(' ') + 1]))
                                tag_counter += token.count(' ') + 1
                        else:
                            tags_proper.append(tags[tag_counter])
                            tag_counter += 1
                    ostream.write(u''.join(['\t'.join(
                        entry) + '\t' + tag + '\n' for entry, tag in zip(entry_list, tags_proper)]) + '\n')
                else:
                    tags = self.tag_lemmatise_sent(totag)
                    tags_proper = []
                    for token in [e[index] for e in entry_list]:
                        if ' ' in token:
                            if len(token) == 1:
                                tags_proper.append([' ', ' '])
                            else:
                                tags_temp = tags[
                                    tag_counter:tag_counter + token.count(' ') + 1]
                                tag = ' '.join([e[0] for e in tags_temp])
                                lemma = ' '.join([e[1] for e in tags_temp])
                                tags_proper.append([tag, lemma])
                                tag_counter += token.count(' ') + 1
                        else:
                            tags_proper.append(tags[tag_counter])
                            tag_counter += 1
                    ostream.write(''.join(['\t'.join(entry) + '\t' + tag[0] + '\t' + tag[
                                  1] + '\n' for entry, tag in zip(entry_list, tags_proper)]) + '\n')
                entry_list = []
            else:
                entry_list.append(line[:-1].split('\t'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Tagger and lemmatiser for Slovene, Croatian and Serbian')
    parser.add_argument('lang', help='language of the text',
                        choices=['sl', 'sl.ns', 'sl.ns.true', 'sl.ns.lower', 'hr', 'sr'])
    parser.add_argument(
        '-l', '--lemmatise', help='perform lemmatisation as well', action='store_true')
    parser.add_argument(
        '-i', '--index', help='index of the column to be processed', type=int, default=0)
    args = parser.parse_args()

    reldiTagger = ReldiTagger3()
    reldiTagger.load_models(args.lang, lemmatise=args.lemmatise)

    reldiTagger.read_and_write(sys.stdin, args.index - 1, sys.stdout)
