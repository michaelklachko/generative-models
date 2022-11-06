import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import print_debug


class Classifier(nn.Module):
    def __init__(self, args=None) -> None:
        super().__init__()
        self.debug = args.debug
        self.latent_size = args.latent_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(in_features=256*4*4, out_features=self.latent_size)
        self.fc_out = nn.Linear(in_features=self.latent_size, out_features=10)
        # self.fc1 = nn.Linear(in_features=128*8*8, out_features=512)
        # self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.bn4 = nn.BatchNorm1d(num_features=self.latent_size)
        # self.bn5 = nn.BatchNorm1d(num_features=512)
        self.dropout = nn.Dropout(p=args.dropout, inplace=False)
        self.act = args.act_fn
        
    def forward(self, x):
        print_debug(x, self.debug, name='\nClassifier: input')
        x = self.act(self.bn1(self.conv1(x)))
        print_debug(x, self.debug, name='Classifier: after conv1')
        x = self.max_pool(x)
        print_debug(x, self.debug, name='Classifier: after maxpool')
        x = self.act(self.bn2(self.conv2(x)))
        print_debug(x, self.debug, name='Classifier: after conv2')
        x = self.max_pool(x)
        print_debug(x, self.debug, name='Classifier: after maxpool')
        
        x = self.act(self.bn3(self.conv3(x)))
        print_debug(x, self.debug, name='Classifier: after conv3')
        x = self.max_pool(x)
        print_debug(x, self.debug, name='Classifier: after maxpool')
        x = x.view(-1, 4*4*256)
        print_debug(x, self.debug, name='Classifier: after reshape')
        x = self.act(self.bn4(self.fc(x)))
        print_debug(x, self.debug, name='Classifier: after fc')
        x = self.dropout(x)
        self.latent_vector = x
        x = self.fc_out(x)
        print_debug(x, self.debug, name='Classifier: after fc_out')

        # x = x.view(-1, 8*8*128)
        # print_debug(x, self.debug, name='Classifier: after reshape')
        # x = self.act(self.bn5(self.fc1(x)))
        # print_debug(x, self.debug, name='Classifier: after fc1')
        # x = self.dropout(x)
        # if not self.train:
        #     self.features = x
        # x = self.fc2(x)
        # print_debug(x, self.debug, name='Classifier: after fc2')
        
        return x
    
    
    
def compute_accuracy(outputs, labels):
    with torch.no_grad():
        batch_size = labels.size(0)
        pred = outputs.max(1)[1]
        acc = pred.eq(labels).sum().item() * 100.0 / batch_size
        return acc
    
 
def train_classifier(args, classifier=None, train_dataloader=None, test_dataloader=None, device=None):
    # python autoencoder.py --lr 0.001 --wd 0.1 --epochs 100 --dropout 0 --act relu
    print(f'\n\nTraining CIFAR-10 Classifier\n\n')
    
    if classifier is None:
        classifier = Classifier(args=args).to(device)
    optim = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=args.epochs * len(train_dataloader))

    classifier.train()
    for epoch in range(args.epochs):
        classifier.train()
        tr_accs = []
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            loss = loss_fn(outputs, labels)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_scheduler.step()
            
            tr_acc = compute_accuracy(outputs, labels)
            tr_accs.append(tr_acc)
        
        train_acc = np.mean(tr_accs)
        test_acc = evaluate_classifier(classifier, test_dataloader=test_dataloader, device=device)
            
        print(f'Epoch {epoch}  train {train_acc:.2f}  test {test_acc:.2f}  LR {lr_scheduler.get_last_lr()[0]:.4f}')
    
    checkpoint_path = f'checkpoints/cifar_classifier_{args.latent_size}.pth'
    print(f'\n\nSaving classifier model checkpoint to {checkpoint_path}')
    torch.save(classifier, checkpoint_path)
    # torch.save(classifier, f'checkpoints/{args.tag}cifar_classifier_{args.latent_size}.pth')
    return classifier

def evaluate_classifier(classifier, test_dataloader=None, device=None):
    te_accs = []
    classifier.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            te_acc = compute_accuracy(outputs, labels)
            te_accs.append(te_acc)
    return np.mean(te_accs)