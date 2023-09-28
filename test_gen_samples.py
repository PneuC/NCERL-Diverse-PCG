import json
import argparse
import time

import numpy as np
from tests import evaluate_rewards, evaluate_mpd
from src.smb.level import load_batch
from src.utils.filesys import getpath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--parallel', type=int, default=50)
    parser.add_argument('--rfunc', type=str)
    args = parser.parse_args()
    start = time.time()
    lvls = load_batch(getpath(args.path, 'samples.lvls'))
    rewards = [sum(item) for item in evaluate_rewards(lvls, args.rfunc, parallel=args.parallel)]
    diversity = evaluate_mpd(lvls)
    with open(getpath(args.path, 'performance.csv'), 'w') as f:
        json.dump({'reward': np.mean(rewards), 'diversity': diversity}, f)
    print(f'Evaluation for {args.path} finished,', '%.2f' % (time.time() - start))

    pass
