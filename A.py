from tqdm import tqdm
import numpy as np
import torch, medmnist, sys
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from medmnist import INFO, Evaluator

received_var=sys.argv[1]

# CNN model
class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),nn.BatchNorm2d(128))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1152, 256)  # Adjusted the input size based on the modified structure
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print("After conv1:", x.size())
        x = self.pool(x)
        #print("After pool1:", x.size())
        x = F.relu(self.conv2(x))
        #print("After conv2:", x.size())
        x = self.pool(x)
        #print("After pool2:", x.size())
        x = F.relu(self.conv3(x))
        #print("After conv3:", x.size())
        x = self.pool(x)
        #print("After pool3:", x.size())
        x = x.view(x.size(0), -1)
        #print("After flattening:", x.size())
        x = F.relu(self.fc1(x))
        #print("After fc1:", x.size())
        x = self.fc2(x)
        #print("After fc2:", x.size())
        return x
            
"""def train_and_evaluate(received_var, num_epochs, batch_size, lr):
    
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    n_channels, n_classes = info['n_channels'] , len(info['label']) # '1''2'
    DataClass = getattr(medmnist, info['python_class'])   #'<class 'medmnist.dataset.PneumoniaMNIST'>'

    avg_train_accuracy = 0.0
    avg_val_accuracy = 0.0
    avg_test_accuracy = 0.0

    num_runs=3
    for run in range(num_runs):
        NUM_EPOCHS = num_epochs
        BATCH_SIZE = batch_size
        lr = lr

        # preprocessing
        data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])

        # load the data
        train_dataset = DataClass(split='train', transform=data_transform, root=received_var)    
        val_dataset = DataClass(split='val', transform=data_transform, root=received_var)       
        test_dataset = DataClass(split='test', transform=data_transform, root=received_var)      

        # encapsulate data into dataloader form
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

        #print(train_dataset)
        #print("===================")
        #print(val_dataset)
        #print("===================")
        #print(test_dataset)

        model = Net(in_channels=n_channels, num_classes=n_classes)
        # define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # train
        for epoch in range(NUM_EPOCHS):            
            model.train()
            for inputs, targets in tqdm(train_loader):
                # forward + backward + optimize
                optimizer.zero_grad()
                outputs = model(inputs)
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()


        # evaluation
        print(f'==> Evaluating - Run {run + 1} ...')
        train_accuracy = test(model,train_loader,data_flag,received_var,'train')
        val_accuracy = test(model,val_loader,data_flag,received_var,'val')
        test_accuracy = test(model,test_loader,data_flag,received_var,'test')

        avg_train_accuracy += train_accuracy
        avg_val_accuracy += val_accuracy
        avg_test_accuracy += test_accuracy
        
    avg_train_accuracy /= num_runs
    avg_val_accuracy /= num_runs
    avg_test_accuracy /= num_runs

    print(f'Average Train Accuracy: {avg_train_accuracy}, Average Validation Accuracy: {avg_val_accuracy}, Average Test Accuracy: {avg_test_accuracy}')


    return avg_train_accuracy, avg_val_accuracy, avg_test_accuracy  # Return the test accuracy




def find_best_hyperparameters(received_var, num_epochs_list, batch_size_list, lr_list):
    best_avg_test_accuracy = 0.0
    best_hyperparameters = {}

    for num_epochs in num_epochs_list:
        for batch_size in batch_size_list:
            for lr in lr_list:
                print(f'Training with Num_Epochs={num_epochs}, Batch_Size={batch_size}, LR={lr}')
                avg_train_accuracy, avg_val_accuracy, avg_test_accuracy = train_and_evaluate(received_var, num_epochs, batch_size, lr)
                print('=' * 30)

                # Update best hyperparameters if the current test accuracy is better
                if avg_test_accuracy > best_avg_test_accuracy:
                    
                    best_avg_train_accuracy= avg_train_accuracy
                    best_avg_val_accuracy= avg_val_accuracy

                    best_avg_test_accuracy = avg_test_accuracy
                    best_hyperparameters = {'Num_Epochs': num_epochs, 'Batch_Size': batch_size, 'LR': lr}

    print(f'Best Hyperparameters: {best_hyperparameters}')
    print(f'Best Average Train Accuracy: {best_avg_train_accuracy}, Val Accuracy: {best_avg_val_accuracy}, Test Accuracy: {best_avg_test_accuracy}')


#test parameters:
num_epochs_list = [3, 4, 5]
batch_size_list = [64 ,128, 256]
lr_list = [0.001, 0.01, 0.1]

find_best_hyperparameters(received_var, num_epochs_list, batch_size_list, lr_list)

"""

def test(model, data_loader, data_flag, received_var, split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split,root=received_var)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
        return metrics[1]

train_accuracies = []
val_accuracies = []
test_accuracies = []
losses = []

#final
data_flag = 'pneumoniamnist'
info = INFO[data_flag]
n_channels, n_classes = info['n_channels'] , len(info['label']) # '1''2'
DataClass = getattr(medmnist, info['python_class'])   #'<class 'medmnist.dataset.PneumoniaMNIST'>'
data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, root=received_var)    
val_dataset = DataClass(split='val', transform=data_transform, root=received_var)       
test_dataset = DataClass(split='test', transform=data_transform, root=received_var)      

#load best parameters
NUM_EPOCHS,BATCH_SIZE,lr=10,64,0.01

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


"""
model = Net(in_channels=n_channels, num_classes=n_classes)
# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# train
for epoch in range(NUM_EPOCHS):            
    model.train()
    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.squeeze().long()
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f'==> Evaluating - Epoch {epoch + 1} ...')
    train_accuracy = test(model, train_loader, data_flag, received_var, 'train')
    val_accuracy = test(model, val_loader, data_flag, received_var, 'val')
    test_accuracy = test(model, test_loader, data_flag, received_var, 'test')

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    test_accuracies.append(test_accuracy)

torch.save(model.state_dict(), 'A_trained_model.pth')

# Plotting the learning curve
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label='Validation Accuracy')
plt.plot(range(1, NUM_EPOCHS + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.show()   
"""



# Create an instance of your model
model = Net(in_channels=n_channels, num_classes=n_classes)

# Load the trained model state dictionary
model.load_state_dict(torch.load('A_trained_model.pth'))

# Set the model to evaluation mode
model.eval()
# evaluation
print(f'==> Evaluating ...')
train_accuracy = test(model,train_loader,data_flag,received_var,'train')
val_accuracy = test(model,val_loader,data_flag,received_var,'val')
test_accuracy = test(model,test_loader,data_flag,received_var,'test')


