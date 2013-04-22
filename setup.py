from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = "machine-learning",
    version = "0.1",
    packages=['machine_learning', 'machine_learning.external_code', 'machine_learning.data', 'machine_learning.external_code.fisher'],
    package_data = {
        '': ['*.txt', '*.rst', '*.p'],
        },
    author = "Vik Paruchuri",
    author_email = "vik@edx.org",
    description = "Machine learning based automated text classification for essay scoring.",
    license = "AGPL",
    keywords = "ml machine learning nlp essay education",
    url = "https://github.com/edx/machine-learning",
    include_package_data = True,
    )