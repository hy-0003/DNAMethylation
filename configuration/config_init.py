import os
import sys
import argparse
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


def get_config():
    parse = argparse.ArgumentParser(description='common train config')

    parse.add_argument('-dataset-name', type=str, default='5hmC_H.sapiens', 
                        help='DataSet Name(dataset_paths:key name)|数据集名称(对应dataset_paths中的key)')
    
    # Choose the dataset name to default
    # 5hmC DataSet    '5hmC_H.sapiens','5hmC_M.musculus'
    # 4mC  DataSet    '4mC_C.equisetifolia','4mC_F.vesca','4mC_S.cerevisiae','4mC_Tolypocladium'
    # 6mA  DataSet    '6mA_A.thaliana''6mA_C.elegans''6mA_C.equisetifolia''6mA_D.melanogaster'
    # 6mA  DataSet    '6mA_F.vesca''6mA_H.sapiens''6mA_R.chinensis''6mA_S.cerevisiae''6mA_T.thermophile''6mA_Tolypocladium''6mA_Xoc BLS256'
    
    parse.add_argument('-path-save', type=str, default='result/', help='The save position|保存的位置')
    parse.add_argument('-save-best', type=bool, default=True, help='Save the best Weight|当得到更好的准确度是否要保存')
    parse.add_argument('-cuda', type=bool, default=True)
    # parse.add_argument('-cuda', type=bool, default=False)
    parse.add_argument('-device', type=int, default=0)
    parse.add_argument('-num_workers', type=int, default=0)
    parse.add_argument('-num_class', type=int, default=2)

    # save path
    parse.add_argument('-train-name', type=str, help='-train-name|训练名称')
    parse.add_argument('-test-name', type=str, help='-test-name|测试名称')
    parse.add_argument('-path_train_data', type=str, default='', help='TrainData path|训练数据路径')
    parse.add_argument('-path_test_data', type=str, default='', help='TestData path|测试数据路径')
    parse.add_argument('-path-params', type=str, default=None, help='Params path|模型参数路径')
    parse.add_argument('-model-save-name', type=str, default='BERT', help='Save weights Name|保存模型的命名')
    parse.add_argument('-save-figure-type', type=str, default='png', help='Save figure type|保存图片的文件类型')


    # training
    parse.add_argument('-interval-log', type=int, default=10, help='Logging frequency (in batches)|经过多少batch记录一次训练状态')
    parse.add_argument('-interval-test', type=int, default=1, help='Evaluation frequency (in epochs)|经过多少epoch对测试集进行测试')
    parse.add_argument('-optimizer', type=str, default='AdamW', help='Optimizer name|优化器名称')
    parse.add_argument('-loss-func', type=str, default='CE', help='Loss function name|损失函数名称')


    # TODO  decide learn name, adversarial, threshold, kmers, epoch
    parse.add_argument('-learn-name', type=str, default='TrainResult', help='The Name of this Training Run|本次训练名称')
    parse.add_argument('-adversarial', type=bool, default=True)
    # parse.add_argument('-adversarial', type=bool, default=False)
    parse.add_argument('-threshold', type=float, default=0.95, help='Accuracy Threshold|准确率阈值,当准确率超过此值将进行保存')
    parse.add_argument('-kmers',type=int, default=[6, 0])
    parse.add_argument('-kmer',type=int, default=0)
    parse.add_argument('-epoch', type=int, default=1, help='Epoch|迭代次数')

    
    # TODO  change batchsize, lr, reg, b
    parse.add_argument('-batch-size', type=int, default=64)
    parse.add_argument('-lr', type=float, default=0.00001)
    parse.add_argument('-reg', type=float, default=0.003, help='weight_decay')
    parse.add_argument('-b', type=float, default=0.06, help='flood level')


    config = parse.parse_args()

    config.dataset_paths = {
        # 5hmC
        '5hmC_H.sapiens': {
            'train': 'data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/train.tsv',
            'test': 'data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/test.tsv'
        },
        '5hmC_M.musculus': {
            'train': 'data/DNA_MS/tsv/5hmC/5hmC_M.musculus/train.tsv',
            'test': 'data/DNA_MS/tsv/5hmC/5hmC_M.musculus/test.tsv'
        },
        # 4mC
        '4mC_C.equisetifolia': {
            'train': 'data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/train.tsv',
            'test': 'data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/test.tsv'
        },
        '4mC_F.vesca': {
            'train': 'data/DNA_MS/tsv/4mC/4mC_F.vesca/train.tsv',
            'test': 'data/DNA_MS/tsv/4mC/4mC_F.vesca/test.tsv'
        },
        '4mC_S.cerevisiae': {
            'train': 'data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/train.tsv',
            'test': 'data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/test.tsv'
        },
        '4mC_Tolypocladium': {
            'train': 'data/DNA_MS/tsv/4mC/4mC_Tolypocladium/train.tsv',
            'test': 'data/DNA_MS/tsv/4mC/4mC_Tolypocladium/test.tsv'
        },
        # 6mA
        '6mA_A.thaliana': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_A.thaliana/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_A.thaliana/test.tsv'
        },
        '6mA_C.elegans': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_C.elegans/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_C.elegans/test.tsv'
        },
        '6mA_C.equisetifolia': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/test.tsv'
        },
        '6mA_D.melanogaster': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_D.melanogaster/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_D.melanogaster/test.tsv'
        },
        '6mA_F.vesca': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_F.vesca/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_F.vesca/test.tsv'
        },
        '6mA_H.sapiens': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_H.sapiens/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_H.sapiens/test.tsv'
        },
        '6mA_R.chinensis': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_R.chinensis/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_R.chinensis/test.tsv'
        },
        '6mA_S.cerevisiae': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/test.tsv'
        },
        '6mA_T.thermophile': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_T.thermophile/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_T.thermophile/test.tsv'
        },
        '6mA_Tolypocladium': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_Tolypocladium/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_Tolypocladium/test.tsv'
        },
        '6mA_Xoc BLS256': {
            'train': 'data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/train.tsv',
            'test': 'data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/test.tsv'
        }
    }

    
    if not config.path_train_data or not config.path_test_data:
        if config.dataset_name not in config.dataset_paths:
            raise ValueError(f"Data Name Error:'{config.dataset_name}' is not define in dataset_paths")
        
        config.path_train_data = os.path.join(rootPath, config.dataset_paths[config.dataset_name]['train'])
        config.path_test_data = os.path.join(rootPath, config.dataset_paths[config.dataset_name]['test'])
        
    del config.dataset_paths
    return config