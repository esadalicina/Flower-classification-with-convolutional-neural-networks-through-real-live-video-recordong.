# Imports for the code
import os
import random
import shutil
import numpy as np
from fastai.basic_train import load_learner
from fastai.metrics import error_rate
from fastai.train import ClassificationInterpretation
from fastai.vision import ImageDataBunch, get_transforms, plt, imagenet_stats, cnn_learner, open_image
from torchvision.models import resnet34
import warnings

warnings.simplefilter('ignore')


# Directories

data_path = 'C:/Users/Esada Licina/UNI.lu BICS/BICS sem2/Technic Part BSP2/BSP2-Technical_part_Python/Flower'
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
train_valid_path = 'C:/Users/Esada Licina/UNI.lu BICS/BICS sem2/Technic Part BSP2/BSP2-Technical_part_Python/Classification'

# % of train/valid data

train_ratio = 0.8
valid_ratio = 0.2


# Create the train/valid directories with our dataset


def train_valid_split():
    for dataset in categories:

        if not os.path.exists(train_valid_path + '/train_set'):
            os.mkdir(train_valid_path + '/train_set')
            os.mkdir(train_valid_path + '/valid_set')

        if not os.path.exists(train_valid_path + '/train_set/' + dataset):
            os.mkdir(train_valid_path + '/train_set/' + dataset)
            os.mkdir(train_valid_path + '/valid_set/' + dataset)
        else:
            for dir_train in os.listdir(train_valid_path + '/train_set/' + dataset):
                os.remove(train_valid_path + '/train_set/' + dataset + '/' + dir_train)
            for dir_test in os.listdir(train_valid_path + '/valid_set/' + dataset):
                os.remove(train_valid_path + '/valid_set/' + dataset + '/' + dir_test)

        images = os.path.join(data_path, dataset)
        random_images = random.sample(os.listdir(images), 200)

        train_folder, valid_folder = np.split(np.array(random_images), [int(len(random_images) * train_ratio)])
        valid_folder, train_folder = np.split(np.array(random_images), [int(len(random_images) * valid_ratio)])

        train_folder = [images + '/' + name for name in train_folder.tolist()]
        valid_folder = [images + '/' + name for name in valid_folder.tolist()]

        for name in train_folder:
            shutil.copy(name, train_valid_path + '/train_set/' + dataset)

        for name in valid_folder:
            shutil.copy(name, train_valid_path + '/valid_set/' + dataset)


# Loading our data with ImageDataBunch

def data_var():
    np.random.seed(42)
    data = ImageDataBunch.from_folder(train_valid_path, train='.', valid_pct=0.2,
                                      ds_tfms=get_transforms(), size=224, num_workers=4)
    data.normalize(imagenet_stats)

    return data


def data_bunch():
    str(data_var())
    print(data_var().classes)
    print(data_var())

    data_var().show_batch(rows=3, figsize=(6, 7))
    plt.show()


# Creating and loading our model

def learn_var():
    learn = cnn_learner(data_var(), resnet34, metrics=error_rate, pretrained=True)
    return learn


def learn_step():
    learn_var().fit_one_cycle(50)

    interp = ClassificationInterpretation.from_learner(learn_var())
    interp.plot_confusion_matrix()
    plt.show()

    interp.plot_top_losses(9, figsize=(15, 15))
    plt.show()


def load_step(images):
    export = "C:/Users/Esada Licina/UNI.lu BICS/BICS sem2/Technic Part BSP2/BSP2-Technical_part_Python/Classification/export/"
    learn=load_learner(export)
    pred=learn.predict(images)
    print('This flower belongs to the class', pred[0])
    print(pred[2])


# Test the above load_step function by unknown images
if __name__ == "__main__":

    new_data_path = "C:/Users/Esada Licina/UNI.lu BICS/BICS sem2/Technic Part BSP2/BSP2-Technical_part_Python/New_Flowers/"
    new_images = os.path.join(new_data_path)
    for single_img in os.listdir(new_images):
        print(single_img)
        img = open_image(new_data_path + single_img)
        load_step(images=img)


