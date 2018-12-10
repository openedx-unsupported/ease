import time
import sys
import random
import pickle
from ease.create import create

MIN_SCORE = 0
MAX_SCORE = 5
NUM_LINES_PER_ESSAY = 100
NUM_TRIALS = 5


def main(num_essays):
    essays = []
    print "Loading essays..."
    with open('essaycorpus.txt') as corpus:
        lines = []
        for line in range(NUM_LINES_PER_ESSAY):
            line = corpus.readline()
            if line == '':
                corpus.seek(0)
            lines.append(line)

        for _ in range(num_essays):
            essays.append(''.join(lines))

    scores = [
        random.randint(MIN_SCORE, MAX_SCORE)
        for _ in range(num_essays)
    ]

    print "Training model..."
    times = []
    for trial in range(NUM_TRIALS):
        print "-- trial #{}".format(trial)
        start = time.time()
        results = create(essays, scores, "")
        end = time.time()
        times.append(end - start)
        print "-- took {} seconds".format(times[-1])

    avg = sum(times) / len(times)
    variance = sum((avg - t) ** 2 for t in times) / len(times)
    print "== Took {} seconds (avg) with {} variance to train on {} essays".format(
        avg, variance, num_essays
    )

    if not results['success']:
        print "Errors occurred!"
        for error in results['errors']:
            print error
    else:
        print "Saving model..."
        with open('feature_extractor', 'w') as feature_ext_file:
            pickle.dump(results['feature_ext'], feature_ext_file)
        with open('model', 'w') as model_file:
            pickle.dump(results['classifier'], model_file)


if __name__ == "__main__":
    main(int(sys.argv[1]))
