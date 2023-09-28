import time
import argparse
from tests import evaluate_rewards, evaluate_gen_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--parallel', type=int, default=50)
    parser.add_argument('--rfunc', type=str)
    args = parser.parse_args()
    start = time.time()
    evaluate_gen_log(args.path, args.rfunc, parallel=args.parallel)
    print(f'Evaluation for {args.path} finished,', '%.2f' % (time.time() - start))
    pass
