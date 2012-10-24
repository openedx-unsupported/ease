# Not django (for now), but use the same settings format anyway

import json
import os
from path import path
import sys

ROOT_PATH = path(__file__).dirname()
REPO_PATH = ROOT_PATH
ENV_ROOT = REPO_PATH.dirname()

# DEFAULTS

DEBUG = False

# Must end in '/'
RUN_URL = 'http://127.0.0.1:3031/'  # Victor's VM ...
RUN_URL = 'http://sandbox-runserver-001.m.edx.org:8080/'
RUN_URL = 'http://sandbox-runserver.elb.edx.org:80/'

GRADER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

# AWS

if os.path.isfile(ENV_ROOT / "env.json"):
    print "Opening env.json file"
    with open(ENV_ROOT / "env.json") as env_file:
        ENV_TOKENS = json.load(env_file)

    RUN_URL = ENV_TOKENS['RUN_URL']

    LOG_DIR = ENV_TOKENS['LOG_DIR']

    # Should be absolute path to 6.00 grader dir.
    # NOTE: This means we only get one version of 6.00 graders available--has to
    # be the same for internal and external class.  Not critical -- can always
    # use different grader file if want different problems.
    GRADER_ROOT = ENV_TOKENS.get('GRADER_ROOT')
