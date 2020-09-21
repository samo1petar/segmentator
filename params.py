from lib.architecture.Model import Model
from lib.loader.RecordReader import RecordReader
from lib.loader.RecordWriter import RecordWriter
from lib.loss.softmax_cross_entropy import SoftmaxCrossEntropy
from lib.optimizer.adam import optimizer_and_learning_rate
from lib.split.split import Split


params = {
    'batch_size'       : 1,
    'learning_rate'    : 0.0001,
    'decay_steps'      : 2000,
    'decay_rate'       : 0.9,
    'print_every_iter' : 100,
    'eval_every_iter'  : 1000,
    'max_iter'         : 1000000,
    'clip_gradients'   : 2.0,
    'results_dir'      : 'results',
    'name'             : 'first_test',
    'image_size'       : (512, 512),
}


class Definition:

    split = Split(
        images_dir='/media/david/A/Datasets/PlayHippo/images',
        masks_dir='/media/david/A/Datasets/PlayHippo/masked_images_cleaned',
    )

    train_set, test_set = split.create_splits()

    writer = RecordWriter(
        data_path='/media/david/A/Dataset/PlayHippo',
        record_dir='records',
        record_name='data',
        train_set=train_set,
        test_set=test_set,
        save_n_test_images=1,
        save_n_train_images=1,
    )

    reader = RecordReader(
        record_dir='records',
        record_name='data',
        batch_size=1,
        shuffle_buffer=1,
        num_parallel_calls=1,
        num_parallel_reads=1,
        prefatch_buffer_size=1,
        count=-1,
    )

    model = Model(name='Model', M=1)

    loss = SoftmaxCrossEntropy()

    optimizer = optimizer_and_learning_rate(
        learning_rate = params['learning_rate'],
        batch_size    = params['batch_size'],
        decay_steps   = params['decay_steps'],
        decay_rate    = params['decay_rate'],
    )
