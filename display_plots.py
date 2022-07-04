from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from torchvision import models, datasets, transforms
import torch
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToPILImage


def confusion_shit():
    val_dir = os.path.join("data", "val")
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']

    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 10)
    model.load_state_dict(torch.load("trained_model.pt"))
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    with torch.no_grad():
        for inputs, labels in torch.utils.data.DataLoader(datasets.ImageFolder(val_dir, data_transforms),
                                                          batch_size=10_000):
            outputs = model(inputs)
            predict = outputs.argmax(axis=1)
            cm = confusion_matrix(labels, predict)

            cm_df = pd.DataFrame(data=cm, index=classes, columns=classes)
            tpil = ToPILImage()
            for idx in np.argwhere(predict != labels)[0]:
                idx = idx.item()
                print(f'Predicted: {classes[predict[idx].item()]}')
                print(f'True: {classes[labels[idx].item()]}')
                plt.imshow(tpil(inputs[idx]))
                plt.show()
                print()


def plot_PCA():
    train_dir = os.path.join("data", "train")
    # val_dir = os.path.join("data", "val")
    classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']

    x = models.resnet50(pretrained=False)
    x.fc = torch.nn.Linear(2048, 10)
    x.load_state_dict(torch.load("trained_model.pt"))
    model = x  # Sequential(*(list(x.children())[:-1]))
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    with torch.no_grad():
        for inputs, labels in torch.utils.data.DataLoader(datasets.ImageFolder(train_dir, data_transforms),
                                                          batch_size=10_000):
            outputs = model(inputs)
            outputs = outputs.reshape(outputs.shape[0], -1)

            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(outputs)
            principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
            principalDf['target'] = labels
            principalDf['target'] = principalDf['target'].apply(lambda t: classes[t])

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('1st Principal Component', fontsize=15)
            ax.set_ylabel('2nd Principal Component', fontsize=15)
            ax.set_title('2 Component PCA', fontsize=20)
            targets = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for target, color in zip(targets, colors):
                indicesToKeep = principalDf['target'] == target
                ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                           , principalDf.loc[indicesToKeep, 'principal component 2']
                           , c=color
                           , s=50)
            ax.legend(targets)
            ax.grid()
            plt.show()

