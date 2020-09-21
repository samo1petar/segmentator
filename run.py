import argparse
from params import Definition, params
# from lib.execution.eval import eval
from lib.execution.train import train

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        dest    = 'mode',
        type    = str,
        default = 'train',
        help    = 'Choose between train and test',
    )
    parser.add_argument(
        '--no_images',
        dest   = 'no_images',
        action = 'store_false',
        help   = 'In eval mode, wrong images won\'t be saved',
    )
    args = parser.parse_args()

    if args.mode == 'train':
        train(
            model            = Definition.model,
            loader           = Definition.reader,
            loss_object      = Definition.loss,
            optimizer        = Definition.optimizer,
            print_every_iter = params['print_every_iter'],
            eval_every_iter  = params['eval_every_iter'],
            max_iter         = params['max_iter'],
            clip_gradients   = params['clip_gradients'],
            results_dir      = params['results_dir'],
            name             = params['name'],
        )
    elif args.mode in ['test', 'eval']:
        eval(
            model       = Definition.model,
            loader      = Definition.reader,
            results_dir = params['results_dir'],
            save_images = args.no_images,
        )

if __name__ == '__main__':
    run()
