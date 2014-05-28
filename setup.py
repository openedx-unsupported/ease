from setuptools import setup, find_packages


def is_requirement(line):
    """
    Return True if the requirement line is a package requirement;
    that is, it is not blank, a comment, or editable.
    """
    # Remove whitespace at the start/end of the line
    line = line.strip()

    # Skip blank lines, comments, and editable installs
    return not (
        line == '' or
        line.startswith('-r') or
        line.startswith('#') or
        line.startswith('-e') or
        line.startswith('git+')
    )


REQUIREMENTS = [
    line.strip() for line in (
        open("pre-requirements.txt").readlines() +
        open("requirements.txt").readlines() +
        open("base_requirements.txt").readlines()
    )
    if is_requirement(line)
]


setup(
    name = "ease",
    version = "0.1.2",
    packages=['ease', 'ease.external_code', 'ease.data', 'ease.data.nltk_data'],
    package_data = {
        '': ['*.txt', '*.rst', '*.p', '*.zip'],
        },
    author = "Vik Paruchuri",
    author_email = "vik@edx.org",
    description = "Machine learning based automated text classification library.  Useful for essay scoring and other tasks.  Please see https://github.com/edx/discern for an API wrapper of this code.",
    license = "AGPL",
    keywords = "ml machine learning nlp essay education",
    url = "https://github.com/edx/ease",
    include_package_data = True,
    install_requires=REQUIREMENTS,
)
