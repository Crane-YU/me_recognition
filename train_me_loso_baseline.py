import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.optim import Adam, lr_scheduler
from capsule.data import data_split, sample_data
from torchvision import transforms
from capsule.data import get_meta_data, Dataset
from capsule.evaluations import Meter
from tqdm import tqdm
from sklearn.utils import shuffle
from torchvision import models
from torch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss()
device = torch.device("cuda:0")


# VGG Baseline
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = models.vgg11(pretrained=True)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=3)

    def forward(self, x):
        output = F.softmax(self.model(x), dim=-1)
        return output


# ResNet Baseline
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=3)

        for module in ['conv1', 'bn1', 'layer1']:
            for param in getattr(self.model, module).parameters():
                param.requires_grad = False

    def forward(self, x):
        output = F.softmax(self.model(x), dim=-1)
        return output


data_apex_frame_path = 'datasets/data_apex.csv'
data_four_frames_path = 'datasets/data_four_frames.csv'
data_root = '/home/ubuntu/Datasets/MEGC/process/'
batch_size = 32
lr = 0.0001
lr_decay_value = 0.9
num_classes = 3
epochs = 30

x_meter = Meter()
batches_scores = []


def load_me_data(data_root, file_path, subject_out_idx, batch_size=32, num_workers=4):
    df_train, df_val = data_split(file_path, subject_out_idx)
    df_four = pd.read_csv(data_four_frames_path)
    df_train_sampled = sample_data(df_train, df_four)
    df_train_sampled = shuffle(df_train_sampled)

    train_paths, train_labels = get_meta_data(df_train_sampled)

    train_transforms = transforms.Compose([transforms.Resize((234, 240)),
                                           transforms.RandomRotation(degrees=(-8, 8)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                  saturation=0.2, hue=0.2),
                                           transforms.RandomCrop((224, 224)),
                                           transforms.ToTensor()])

    train_dataset = Dataset(root=data_root,
                            img_paths=train_paths,
                            img_labels=train_labels,
                            transform=train_transforms)

    val_transforms = transforms.Compose([transforms.Resize((234, 240)),
                                         transforms.RandomRotation(degrees=(-8, 8)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor()])

    val_paths, val_labels = get_meta_data(df_val)
    val_dataset = Dataset(root=data_root,
                          img_paths=val_paths,
                          img_labels=val_labels,
                          transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False)
    return train_loader, val_loader


def on_epoch(model, optimizer, lr_decay, train_loader, test_loader, epoch):
    # Training mode
    model.train()
    lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
    train_loss = 0.0
    correct = 0.
    meter = Meter()

    steps = len(train_loader.dataset) // batch_size + 1
    with tqdm(total=steps) as progress_bar:
        for i, (x, y) in enumerate(train_loader):  # batch training
            y = F.one_hot(y, num_classes)
            x, labels = x.to(device), y.to(device)  # GPU mode

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred = model(x)  # forward
            _, y_pred = torch.max(y_pred.data, 1)

            loss = criterion(y_pred, labels)
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            train_loss += loss.item() * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

            y_pred = y_pred.data.max(1)[1]

            meter.add(labels.cpu().numpy(), y_pred.cpu().numpy())
            correct += (y_pred == y).sum().item()

            progress_bar.set_postfix(loss=loss.item(), correct=correct)
            progress_bar.update(1)

        train_loss /= float(len(train_loader.dataset))
        train_acc = float(correct) / float(len(train_loader.dataset))
        scores_train = meter.value()
        meter.reset()
        print('Training UAR: %.4f' % (scores_train[0].mean()), scores_train[0])
        print('Training UF1: %.4f' % (scores_train[1].mean()), scores_train[1])

    correct = 0
    test_loss = 0.0

    model.eval()
    for idx, (x, y) in enumerate(test_loader):  # batch testing
        y = F.one_hot(y, num_classes)
        x, y = x.to(device), y.to(device)  # convert input data to GPU Variable

        y_pred = model(x)  # forward
        _, y_pred = torch.max(y_pred.data, 1)

        loss = criterion(y_pred, y)  # compute loss
        test_loss += loss.item() * x.size(0)  # record the batch loss

        y_pred = y_pred.data.max(1)[1]

        meter.add(y.cpu().numpy(), y_pred.cpu().numpy())
        correct += (y_pred == y).sum().item()

        if (epoch + 1) % 2 == 0 and idx % steps == 0:
            print('y_true\n', y[:30])
            print('y_pred\n', y_pred[:30])
            print('y_true', y.sum(dim=0))

    scores_test = meter.value()
    print('y_true', y.sum(dim=0))
    print('Testing UAR: %.4f' % (scores_test[0].mean()), scores_test[0])
    print('Testing UF1: %.4f' % (scores_test[1].mean()), scores_test[1])

    test_loss /= float(len(test_loader.dataset))
    test_acc = float(correct) / float(len(test_loader.dataset))
    return train_loss, train_acc, test_loss, test_acc, meter


def train_eval(subject_out_idx):
    best_val_uf1 = 0.0
    best_val_uar = 0.0

    # Model & others
    model = VGG()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_value)

    for epoch in range(epochs):
        train_loader, test_loader = load_me_data(data_root, data_apex_frame_path,
                                                 subject_out_idx=subject_out_idx,
                                                 batch_size=batch_size)

        train_loss, train_acc, test_loss, test_acc, meter = on_epoch(model, optimizer, lr_decay,
                                                                     train_loader, test_loader,
                                                                     epoch)

        print("==> Subject out: %02d - Epoch %02d: loss=%.5f, train_acc=%.5f, val_loss=%.5f, "
              "val_acc=%.4f"
              % (subject_out_idx, epoch, train_loss, train_acc,
                 test_loss, test_acc))

        scores = meter.value()
        if scores[1].mean() >= best_val_uf1:
            best_val_uar = scores[0].mean()
            best_val_uf1 = scores[1].mean()
            x_meter.add(meter.Y_true, meter.Y_pred)

    return best_val_uar, best_val_uf1


for i in range(68):
    step_scores = train_eval(subject_out_idx=i)
    batches_scores.append(step_scores)
    x_scores = x_meter.value()
    print('final uar', x_scores[0], x_scores[0].mean())
    print('final uf1', x_scores[1], x_scores[1].mean())

with open('scores_vgg11_no_macro.pkl', 'wb') as file:
    data = dict(meter=x_meter, batches_scores=batches_scores)
    pickle.dump(data, file)