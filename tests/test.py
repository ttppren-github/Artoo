import argparse

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--train-conf', help='yaml configure file for training')
    argp.add_argument('--evaluate-conf', help='yaml configure file for evaluate')
    args = argp.parse_args()

    if args.train_conf:
        with open(args.train_conf, 'r') as f:
            conf = yaml.load(f)
        train(conf)
    elif args.evaluate_conf:
        with open(args.evaluate_conf, 'r') as f:
            conf = yaml.load(f)
        evaluate(conf)
    else:
        argp.print_usage()

    print('bye bye')
