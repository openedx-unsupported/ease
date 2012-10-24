#!/usr/bin/env python
"""Helper script to run a command on sandbox50"""

import json
import logging
import requests
import os
import os.path
import sys
import time
import settings

run_url = None

log = logging.getLogger(__name__)

def upload(paths):
    """
    Given a list of paths, upload them to the sandbox, and return an id that
    identifies the created directory.
    """
    files = dict( (os.path.basename(f), open(f)) for f in paths)
    return upload_files(files)

def upload_files(files):
    endpoint = settings.RUN_URL + 'upload'
    r = requests.post(endpoint, files=files)

    if r.status_code != requests.codes.ok:
        log.error("Request error: {0}".format(r.text))
        return None

    if r.json is None:
        log.error("sandbox50 /upload failed to return valid json.  Response:" +  r.text)
        return None

    id = r.json.get('id')
    log.debug('Upload_files response: ' + r.text)
    return id

def run(id, cmd):
    # Making run request

    headers = {'content-type': 'application/json'}
    run_args = {'cmd': cmd,
                'sandbox': { 'homedir': id }}

    endpoint = settings.RUN_URL + 'run'
    r = requests.post(endpoint, headers=headers, data=json.dumps(run_args))

    if r.json is None:
        log.error("sandbox50 /run failed to return valid json.  Response:" +  r.text)
        return None

    return r.json

def record_suspicious_submission(msg, code_str):
    """
    Record a suspicious submission:

    TODO: upload to edx-studentcode-suspicious bucket on S3.  For now, just
    logging to avoids need for more config changes (S3 credentials, python
    requirements).
    """
    log.warning('Suspicious code: {0}, {1}'.format(msg, code_str))
    

def sb50_run_code(code):
    """
    Upload passed in code file to the code exec sandbox as code.py, run it.

    Return tuple (stdout, stderr), either of which may be None
    """

    #print "Running code: \n{0}".format(code)

    files = {'code.py': ('code.py', code)}
    start = time.time()
    id = upload_files(files)
    # TODO: statsd
    print "upload took %.03f sec" % (time.time() - start)

    start = time.time()
    r = run(id, '/usr/bin/python code.py')
    print "run took %.03f sec" % (time.time() - start)

    return r['stdout'], r['stderr']
