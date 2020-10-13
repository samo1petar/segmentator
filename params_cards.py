from lib.architecture.ModelCard import Model
from lib.loader.RecordReaderCards import RecordReaderCards
from lib.loader.RecordWriterCards import RecordWriterCards
from lib.loss.softmax import SoftmaxCrossEntropy
from lib.optimizer.adam import optimizer_and_learning_rate


params = {
    'batch_size'       : 4,
    'learning_rate'    : 0.001,
    'decay_steps'      : 2000,
    'decay_rate'       : 0.9,
    'print_every_iter' : 100,
    'eval_every_iter'  : 500,
    'sample_every_iter': 4000,
    'max_iter'         : 1000000,
    'clip_gradients'   : 2.0,
    'results_dir'      : 'results',
    'name'             : 'all_cards',
    'image_size'       : (256, 256),
}


class Definition:

    writer = RecordWriterCards(
        data_path='/media/david/A/Datasets/PlayHippo',
        record_dir='records',
        record_name='cards',
        save_n_test_images=1,
        save_n_train_images=1,
        image_size = (256, 256),
    )

    reader = RecordReaderCards(
        record_dir='records',
        record_name='cards',
        batch_size=params['batch_size'],
        shuffle_buffer=1,
        num_parallel_calls=1,
        num_parallel_reads=1,
        prefatch_buffer_size=1,
        count=-1,
        image_size=(256, 256),
    )

    model = Model(name='Model', M=1)

    loss = SoftmaxCrossEntropy()

    optimizer = optimizer_and_learning_rate(
        learning_rate = params['learning_rate'],
        batch_size    = params['batch_size'],
        decay_steps   = params['decay_steps'],
        decay_rate    = params['decay_rate'],
    )
