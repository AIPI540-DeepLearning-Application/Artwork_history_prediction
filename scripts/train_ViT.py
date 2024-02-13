from datasets import load_dataset
from transformers import ViTImageProcessor

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

from torch.utils.data import DataLoader
import torch

from transformers import ViTForImageClassification

from transformers import TrainingArguments, Trainer

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ViT_fine_tuned:
    '''
    This class is used to train and fine tune a ViT model on the artwork dataset
    '''
    def __init__(self):
        self.data_files="https://huggingface.co/datasets/AIPI540/data_with_aug/resolve/main/data.zip"
        self.load_dataset()
        
    # load dataset
    def load_dataset(self):
        ds = load_dataset("imagefolder", data_files=self.data_files)
        # split up training into training + validation + test
        splits = ds['train'].train_test_split(test_size=0.2)

        train_ds = splits['train']
        self.test_ds = splits['test']

        splits_2 = train_ds.train_test_split(test_size=0.1)

        self.train_ds = splits_2['train']
        self.val_ds = splits_2['test']

    def process_label(self):
        self.id2label = {id:label for id, label in enumerate(self.train_ds.features['label'].names)}
        self.label2id = {label:id for id,label in self.id2label.items()}


    def config_transform(self):
        self.process_label()
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        image_mean, image_std = self.processor.image_mean, self.processor.image_std
        size = self.processor.size["height"]

        normalize = Normalize(mean=image_mean, std=image_std)
        self._train_transforms = Compose(
                [
                    RandomResizedCrop(size),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    normalize,
                ]
            )

        self._val_transforms = Compose(
                [
                    Resize(size),
                    CenterCrop(size),
                    ToTensor(),
                    normalize,
                ]
            )

    def train_transforms(self,examples):
        examples['pixel_values'] = [self._train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    def val_transforms(self,examples):
        examples['pixel_values'] = [self._val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    def apply_transform(self):
        # Set the transforms
        self.config_transform()
        self.train_ds.set_transform(self.train_transforms)
        self.val_ds.set_transform(self.val_transforms)
        self.test_ds.set_transform(self.val_transforms)




    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def train_config(self):
        self.apply_transform()
        self.train_dataloader = DataLoader(self.train_ds, collate_fn=self.collate_fn, batch_size=4)


        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                    id2label=self.id2label,
                                                    label2id=self.label2id)

        metric_name = "F1"

        self.args = TrainingArguments(
            f"test-cifar-10",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=4,
            num_train_epochs=20,
            weight_decay=0.01,
            load_best_model_at_end=True,
            logging_dir='logs',
            remove_unused_columns=False,
        )



    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return dict(accuracy=f1_score(predictions, labels,average='micro'))


    def train(self):
        self.train_config()
        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor,
        )

        self.trainer.train()

    #evalution
    def test(self):
        outputs = self.trainer.predict(self.test_ds)



        y_true = outputs.label_ids
        y_pred = outputs.predictions.argmax(1)

        labels = self.train_ds.features['label'].names
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation=45)

def main():
    ViT = ViT_fine_tuned()
    ViT.train()
    ViT.test()

if __name__ == '__main__':
    main()