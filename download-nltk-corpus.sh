#!/usr/bin/env bash

curl -o /tmp/nltk.tmp.tar.tz http://edx-static.s3.amazonaws.com/nltk/nltk-data-20131113.tar.gz
cd /usr/share && sudo tar zxf /tmp/nltk.tmp.tar.tz
