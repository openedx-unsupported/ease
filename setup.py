import os
import re

from setuptools import find_packages, setup


# UPDATED VIA SEMGREP - if you need to remove/modify this method remove this line and add a comment specifying why.
def load_requirements(*requirements_paths):
    """
    Load all requirements from the specified requirements files.

    Requirements will include any constraints from files specified
    with -c in the requirements files.
    Returns a list of requirement strings.
    """
    requirements = {}
    constraint_files = set()

    # groups "my-package-name<=x.y.z,..." into ("my-package-name", "<=x.y.z,...")
    requirement_line_regex = re.compile(r"([a-zA-Z0-9-_.]+)([<>=][^#\s]+)?")

    def add_version_constraint_or_raise(current_line, current_requirements, add_if_not_present):
        regex_match = requirement_line_regex.match(current_line)
        if regex_match:
            package = regex_match.group(1)
            version_constraints = regex_match.group(2)
            existing_version_constraints = current_requirements.get(package, None)
            # it's fine to add constraints to an unconstrained package, but raise an error if there are already
            # constraints in place
            if existing_version_constraints and existing_version_constraints != version_constraints:
                raise BaseException(f'Multiple constraint definitions found for {package}:'
                                    f' {existing_version_constraints} and {version_constraints}.'
                                    f'Combine constraints into one location with {package}'
                                    f'{existing_version_constraints},{version_constraints}.')
            if add_if_not_present or package in current_requirements:
                current_requirements[package] = version_constraints

    # process .in files and store the path to any constraint files that are pulled in
    for path in requirements_paths:
        with open(path) as reqs:
            for line in reqs:
                if is_requirement(line):
                    add_version_constraint_or_raise(line, requirements, True)
                if line and line.startswith('-c') and not line.startswith('-c http'):
                    constraint_files.add(os.path.dirname(path) + '/' + line.split('#')[0].replace('-c', '').strip())

    # process constraint files and add any new constraints found to existing requirements
    for constraint_file in constraint_files:
        with open(constraint_file) as reader:
            for line in reader:
                if is_requirement(line):
                    add_version_constraint_or_raise(line, requirements, False)

    # process back into list of pkg><=constraints strings
    return [f'{pkg}{version or ""}' for (pkg, version) in sorted(requirements.items())]


# UPDATED VIA SEMGREP - if you need to remove/modify this method remove this line and add a comment specifying why
def is_requirement(line):
    """
    Return True if the requirement line is a package requirement.

    Returns:
        bool: True if the line is not blank, a comment,
        a URL, or an included file
    """
    return line and line.strip() and not line.startswith(('-r', '#', '-e', 'git+', '-c'))

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
