# Optimize selection thresholds for a given model (use F2 score as metric)

import argparse
import json

import torch

sys.path.append(os.path.join('..', 'train'))
from hyper import N_LEVELS, 
from predict_placements import predict_placements

STARTING_THRESHOLD = 0.20

def get_optimal_threshold(level, placement_model, test_iter):
    placement_model.eval()

    threshold = STARTING_THRESHOLD
    f2_score = 0

    last_improved = 0

    # find threshold which maximizes the f2 score
    with torch.no_grad():
        # stop optimizing when haven't improved in 5 changes
        while last_improved < 5 and f2_score < 1:
            for batch in testt_iter:
                pass

    return threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, dest='model_dir', help='path to directory containing model files')
    parser.add_argument('-s', type=str, dest='savefile', default='thresholds.json')

    thresholds = {}
    
    for level in range(N_LEVELS):
        thresholds[level] = get_optimal_threshold(level, placement_model, test_iter)

    with open(args.savefile, 'w') as f:
        f.write(json.dumps(thresholds, indent=2))

if __name__ == '__main__':
    main()