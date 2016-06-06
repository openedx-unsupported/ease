#!/usr/bin/env bash

PYTHON=`which python`
$PYTHON -m nltk.downloader stopwords maxent_treebank_pos_tagger wordnet -d /home/travis/nltk_data
