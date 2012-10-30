# Run with arguments train_file prompt_file model_path to generate a sample model file

import os
import sys
import argparse

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

one_up_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(one_up_path)

import model_creator


def main(argv):
    parser = argparse.ArgumentParser(description="Generate model from test data files")
    parser.add_argument('train_file')
    parser.add_argument('prompt_file')
    parser.add_argument('model_path')

    args = parser.parse_args(argv)

    score, text = model_creator.read_in_test_data(args.train_file)
    prompt_string = model_creator.read_in_test_prompt(args.prompt_file)
    print("data read")
    e_set = model_creator.create_essay_set(text, score, prompt_string)
    print("essay set created")
    feature_ext, classifier = model_creator.extract_features_and_generate_model(e_set)
    print("features pulled out and model generated")
    model_creator.dump_model_to_file(prompt_string, feature_ext, classifier, text, score, args.model_path)
    print("model file written")

if __name__ == "__main__":
    main(sys.argv[1:])


def test_model_creation():
    try:
        score, text = model_creator.read_in_test_data("train.tsv")
        prompt_string = model_creator.read_in_test_prompt("prompt.txt")
        e_set = model_creator.create_essay_set(text, score, prompt_string)
        feature_ext, classifier = model_creator.extract_features_and_generate_model(e_set)
        model_creator.dump_model_to_file(prompt_string, feature_ext, classifier, args.model_path)
        assert True
    except:
        assert False
