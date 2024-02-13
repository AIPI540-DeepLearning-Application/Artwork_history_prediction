from .scripts.train_ViT import ViT_fine_tuned
from .scripts.svm import art_svm

def main():
    print("-----------------------ViT model-----------------------")
    ViT = ViT_fine_tuned()
    ViT.train()
    ViT.test()
    print("-----------------------SVM model-----------------------")
    svm_model=art_svm()
    svm_model.train()

if __name__ == '__main__':
    main()