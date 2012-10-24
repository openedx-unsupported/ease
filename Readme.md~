Project to integrate machine learning based essay scoring with xserver. Aspell must be installed and added to path to run.  numpy, scipy and sklearn also need to be installed.

Runnable files:

1. create_test_models.py 

	Generates test models when used like: python create_test_models.py train_file prompt_file model_path.  Use python create_test_models.py train.tsv prompt.txt models/essay_set_1.txt to generate a model using sample data.

2. pyxserver_wsgi.py

	Starts a server instance that can be sent answers to score.  Calls grade.py to score responses.  Run server with gunicorn -w 4 -b 127.0.0.1:3031 pyxserver_wsgi:application .

3. tests/test.py

	Submits test data found in directories within the tests folder to the xserver and displays results.  See tests/simple_essay for an example of how to format files.  You need payload.json, wrong.txt, and answer.txt to make a test.
