import sys
import time
import pickle
from ease.grade import grade


NUM_TRIALS = 5


def main(submission_path):

    print "Loading submission..."
    with open(submission_path) as sub_file:
        submission = sub_file.read()

    print "Loading model..."
    with open('model') as model_file:
        model = pickle.load(model_file)

    print "Loading feature extractor..."
    with open('feature_extractor') as feature_ext_file:
        feature_ext = pickle.load(feature_ext_file)

    grader_data = {
        'model': model,
        'extractor': feature_ext,
        'prompt': 'Test prompt'
    }

    print "Grading submission..."
    times = []
    for trial in range(NUM_TRIALS):
        print "-- trial #{}".format(trial)
        start = time.time()
        result = grade(grader_data, submission)
        end = time.time()
        times.append(end - start)
        print "-- took {} seconds".format(times[-1])

    avg = sum(times) / len(times)
    variance = sum((avg - t) ** 2 for t in times) / len(times)
    print "== Took {} seconds (avg) with {} variance to grade".format(avg, variance)

    if not result['success']:
        print "Error occurred!"
        for error in result['errors']:
            print error

if __name__ == "__main__":
    main(sys.argv[1])
