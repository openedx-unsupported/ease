from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = "ease",
    version = "0.1",
    packages=['ease', 'ease.external_code', 'ease.data', 'ease.external_code.fisher'],
    package_data = {
        '': ['*.txt', '*.rst', '*.p'],
        },
    author = "Vik Paruchuri",
    author_email = "vik@edx.org",
    description = "Machine learning based automated text classification library.  Useful for essay scoring and other tasks.  Please see https://github.com/edx/discern for an API wrapper of this code.",
    license = "AGPL",
    keywords = "ml machine learning nlp essay education",
    url = "https://github.com/edx/ease",
    include_package_data = True,
    )