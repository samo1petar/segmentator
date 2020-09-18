from lib.architecture.Model import Model
from lib.loader.RecordReader import RecordReader
from lib.loader.RecordWriter import RecordWriter
from lib.loss.softmax_cross_entropy import SoftmaxCrossEntropy
from lib.optimizer.adam import optimizer_and_learning_rate


params = {
    'batch_size'       : 10,
    'learning_rate'    : 0.0001,
    'decay_steps'      : 2000,
    'decay_rate'       : 0.9,
    'print_every_iter' : 100,
    'eval_every_iter'  : 1000,
    'max_iter'         : 1000000,
    'clip_gradients'   : 2.0,
    'results_dir'      : 'results',
    'name'             : 'augmented_2',
    'image_size'       : (512, 512),
}


class Definition:

    writer = RecordWriter(
        data_path   = 'data/augmented_2',
        record_dir  = 'records',
        record_name = 'augmented_2',
        image_size  = params['image_size'],
    )

    reader = RecordReader(
        record_dir  = 'records',
        record_name = 'augmented_2',
        image_size  = params['image_size'],
        batch_size  = params['batch_size'],
    )

    model = Model(name='Model', M=1)

    loss = SoftmaxCrossEntropy()

    optimizer = optimizer_and_learning_rate(
        learning_rate = params['learning_rate'],
        batch_size    = params['batch_size'],
        decay_steps   = params['decay_steps'],
        decay_rate    = params['decay_rate'],
    )
