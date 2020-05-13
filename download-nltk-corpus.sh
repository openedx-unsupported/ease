#!/usr/bin/env bash

PYTHON=`which python`
sudo $PYTHON -m nltk.downloader stopwords maxent_treebank_pos_tagger wordnet punkt -d /usr/local/share/nltk_data
