from setuptools import setup, find_packages


def load_requirements(*requirements_paths):
    """
    Load all requirements from the specified requirements files.
    Returns a list of requirement strings.
    """
    requirements = set()
    for path in requirements_paths:
        with open(path) as reqs:
            requirements.update(
                line.split('#')[0].strip() for line in reqs
                if is_requirement(line.strip())
            )
    return list(requirements)


def is_requirement(line):
    """
    Return True if the requirement line is a package requirement;
    that is, it is not blank, a comment, a URL, or an included file.
    """
    return line and not line.startswith(('-r', '#', '-e', 'git+', '-c'))

setup(
    name = "ease",
    version = "0.1.3",
    packages=['ease'],
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
    install_requires=load_requirements('requirements/production.in'),
    tests_require=load_requirements('requirements/test.in'),
)
