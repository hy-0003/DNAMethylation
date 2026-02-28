import os
import sys
import transformers
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from frame import Learner
from configuration import config_init
transformers.logging.set_verbosity(transformers.logging.ERROR)


def Train(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.adjust_model()
    #learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()


# Train need about 6GB GPU memory
if __name__ == '__main__':
    config = config_init.get_config()
    Train(config)