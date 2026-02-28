import os
import sys
import torch
import transformers
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from frame import Learner
from configuration import config_init
transformers.logging.set_verbosity(transformers.logging.ERROR)


# The class does not matter, just look for  if __name__ == '__main__'
class ReviewerConfig:
    def __init__(self, model_weights_path, test_dataset_path):
        # 1. path set
        self.config = config_init.get_config()
        self.path_params = model_weights_path
        self.path_test_data = test_dataset_path
        self.path_save = 'result/Valid_results'
        # 2. device set
        # You can choose use gpu or cpu by define self.cude, but cpu is very slow
        self.cuda = False                           # CPU
        self.cuda = torch.cuda.is_available()       # GPU
        self.device = 0
        # 3. Init set
        self.learn_name = 'Reviewer_Test_Run'
        self.path_train_data = self.config.path_train_data
        self.batch_size = self.config.batch_size
        self.mode = 'train-test'
        self.kmers = self.config.kmers
        self.kmer = self.kmers[0]
        self.num_class = self.config.num_class
        self.b = self.config.b


def predict_on_new_dataset(model_weights_path, test_dataset_path):
    config = ReviewerConfig(model_weights_path, test_dataset_path)

    if config.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
        print(f"Using GPU: {config.device}")
    else:
        print("Using CPU")

    try:
        learner = Learner.Learner(config)
        learner.setIO()
        learner.setVisualization()
        learner.load_data()
        learner.init_model()
        learner.load_params()
        learner.def_loss_func()
        learner.test_model()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Here you can choose the trained_model and test_data to valid our weight, cross-dataset evaluation, external dataset valid
    trained_model_path = 'result/5hmC_H.sapiens/6mer/BERT, ACC[0.951].pt'
    test_data_path = 'data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/test.tsv'

    # external dataset path
    # test_data_path = 'data/external/test.tsv'

    '''
    If you only want to use cpu to chech if the model and weight can work, please use the example dataset we provide below.
    The result SP 0.0000, AUC nan, MCC 0.0000, because we only choose first 100 sequence in positive sample.
    It will take about 20s.
    '''
    # test_data_path = 'data/DNA_MS/tsv/6mA/6mA_D.melanogaster/example_test.tsv'


    if not os.path.exists(trained_model_path):
        print(f"Error: can not find weight file:{trained_model_path}")
    elif not os.path.exists(test_data_path):
        print(f"Error: can not find test file:{test_data_path}")
    else:
        predict_on_new_dataset(
            model_weights_path=trained_model_path,
            test_dataset_path=test_data_path
        )