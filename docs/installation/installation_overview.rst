===============================================
Installation Overview
===============================================

Notes on how to install:

1. cd DIRECTORY_YOU_INSTALLED_TO.  Make sure that you install to the folder ease!
2. sudo apt-get update
3. sudo apt-get upgrade gcc
4. sudo xargs -a apt-packages.txt apt-get install
5. Activate your virtual env (if you have one)
6. pip install -r pre-requirements.txt
7. pip install -r requirements.txt
8. python setup.py install
9. sudo cp -r ease/data/nltk_data /usr/share/nltk_data
