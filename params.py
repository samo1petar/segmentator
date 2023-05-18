from lib.architecture.Model import Model
from lib.loader.RecordReader import RecordReader
from lib.loader.RecordWriter import RecordWriter
from lib.loss.softmax_cross_entropy import SoftmaxCrossEntropy
from lib.loss.DiceLoss import DiceLoss
from lib.optimizer.adam import optimizer_and_learning_rate
from lib.split.split import Split
from lib.tools.file import choose_one_from_dir, choose_model
import os


params = {
    'batch_size'       : 8,
    'learning_rate'    : 0.0001,
    'decay_steps'      : 5000,
    'decay_rate'       : 0.9,
    'print_every_iter' : 100,
    'eval_every_iter'  : 2000,
    'max_iter'         : 1000000,
    'clip_gradients'   : 2.0,
    'results_dir'      : '/home/petar/Projects/Shapes/models',
    'name'             : 'segmentator_full_gray_rect_oval_smooth_dice',
    'image_size'       : (512, 768),
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
        record_name='segmentator_full_gray_rect_oval_smooth',
        train_set=train_set,
        test_set=test_set,
        save_n_test_images=1,
        save_n_train_images=1,
        image_size=params['image_size'],
    )

    reader = RecordReader(
        record_dir='/home/petar/Projects/Shapes/records',
        record_name='segmentator_full_gray_rect_oval_smooth',
        batch_size=params['batch_size'],
        shuffle_buffer=2,
        num_parallel_calls=2,
        num_parallel_reads=2,
        prefatch_buffer_size=5,
        count=-1,
        image_size=params['image_size'],
    )

    model = Model(name='Model', M=2)

    # experiment = choose_one_from_dir(params['results_dir'])
    # model_dir = os.path.join(experiment, 'model')
    # model_path = choose_model(model_dir)
    # model.load_weights(model_path)

    loss = DiceLoss()

    optimizer = optimizer_and_learning_rate(
        learning_rate = params['learning_rate'],
        batch_size    = params['batch_size'],
        decay_steps   = params['decay_steps'],
        decay_rate    = params['decay_rate'],
    )
