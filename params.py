from lib.architecture.Model import Model
from lib.loader.RecordReader import RecordReader
from lib.loader.RecordWriter import RecordWriter
from lib.loss.softmax_cross_entropy import SoftmaxCrossEntropy
from lib.optimizer.adam import optimizer_and_learning_rate
from lib.split.split import Split


params = {
    'batch_size'       : 1,
    'learning_rate'    : 0.001,
    'decay_steps'      : 1000,
    'decay_rate'       : 0.9,
    'print_every_iter' : 10,
    'eval_every_iter'  : 500,
    'max_iter'         : 1000000,
    'clip_gradients'   : 2.0,
    'results_dir'      : '/home/petar/Projects/Shapes/models',
    'name'             : 'segmentator_20_images_gray_icenet',
    'image_size'       : (1024, 1536),
}


class Definition:

    split = Split(
        images_dir='/home/petar/Projects/Shapes/presentations_png_mmseg 6/images',
        masks_dir='/home/petar/Projects/Shapes/presentations_png_mmseg 6/labels',
    )

    train_set, test_set = split.create_splits()

    writer = RecordWriter(
        data_path='/home/petar/Projects/Shapes/presentations_png_mmseg 6/',
        record_dir='/home/petar/Projects/Shapes/records',
        record_name='segmentator_20_images_gray',
        train_set=train_set,
        test_set=test_set,
        save_n_test_images=1,
        save_n_train_images=1,
        image_size=params['image_size'],
    )

    reader = RecordReader(
        record_dir='/home/petar/Projects/Shapes/records',
        record_name='segmentator_20_images_gray',
        batch_size=params['batch_size'],
        shuffle_buffer=2,
        num_parallel_calls=2,
        num_parallel_reads=2,
        prefatch_buffer_size=5,
        count=-1,
        image_size=params['image_size'],
    )

    model = Model(name='Model', M=1)

    loss = SoftmaxCrossEntropy()

    optimizer = optimizer_and_learning_rate(
        learning_rate = params['learning_rate'],
        batch_size    = params['batch_size'],
        decay_steps   = params['decay_steps'],
        decay_rate    = params['decay_rate'],
    )
