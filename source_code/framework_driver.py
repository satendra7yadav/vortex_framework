from ensembled_model.ensembled_model import *
from vision_framework.VortexDetection import _main
import glob
import argparse


parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-vortex', type=str, nargs=2, action="store", help='A required string positional argument to predict the model')
# parser.add_argument('--predict', type=str, help='A required string positional argument to predict the model')
# parser.add_argument('--train', type=str, help='A required string positional argument to train the model')
# parser.add_argument('--ensemble', type=str, help='A required string positional argument for ensembled model')
# parser.add_argument('--svm', type=str, help='A required string positional argument for svm model')

args = parser.parse_args()
print(args.vortex)
if args.vortex:
    if args.vortex[0]=='train':
        if args.vortex[1]=='ensemble':
            train_dataset_path = 'E:\\group project\\vortex_framework\\data\\training_dataset_path\\vortex_57_train.csv'
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # random forest parameters
            train_hyperparameters['n_estimators']=200
            # svm parameters
            train_hyperparameters['kernel']='poly'
            train_hyperparameters['C']=1
            train_hyperparameters['gamma']='auto'

            model_summary = ensembled_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)

        elif args.vortex[1]=='svm':
            print('code not added yet')
            pass
    elif args.vortex[0]=='predict':

        if args.vortex[1]=='ensemble':
            dataset_path = 'E:\\group project\\vortex_framework\\data\\prediction_dataset_path\\time_step_57.csv'
            model_path = 'E:\\group project\\vortex_framework\\data\\trained_models\\ensembled\\time_step_57.pkl'
            image_path = 'E:\\group project\\vortex_framework\\data\\Data\\img57.png'
            vision_model_path = ''
            no_of_bounding_boxes=_main.CVMain(image_path)
            vortex_core = ensembled_predict_model(dataset_path,model_path,no_of_bounding_boxes)
            print(vortex_core)

        elif args.vortex[1]=='svm':
            print('code not added yet')
            pass
else:
    print('Incorrect argument')
