from .scripts.train_ViT import ViT_fine_tuned
from .scripts.svm import art_svm

def main():
    print("-----------------------ViT model-----------------------")
    ViT = ViT_fine_tuned()
    ViT.train()
    ViT.test()
    
    print("-----------------------SVM model-----------------------")
    '''
        The data for the svm model is not uploaded to the github repo due to size constraints
        if you want to run the model, please download the data from the following link:
        https://huggingface.co/datasets/AIPI540/data_with_aug
        and put it under the root folder
    '''
    svm_model=art_svm()
    svm_model.train()

if __name__ == '__main__':
    main()