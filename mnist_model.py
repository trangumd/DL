import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train(self, learning_rate, batch_size, epochs):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        progress = []   

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 100 == 99:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}, Accuracy: {(correct / total) * 100}%')
                    progress.append({
                        'epoch': epoch + 1,
                        'batch': i + 1,
                        'loss': running_loss / 100,  # Example loss calculation
                        'accuracy': (correct / total) * 100   # Example accuracy calculation
                    })
                    running_loss = 0.0
                    correct = 0 
                    total = 0

        print('Finished Training')
        return progress

    # def evaluate(self):
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,))
    #     ])

    #     testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    #     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for data in testloader:
    #             images, labels = data
    #             outputs = self(images)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #     accuracy = 100 * correct / total
    #     print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)
    #     return accuracy


