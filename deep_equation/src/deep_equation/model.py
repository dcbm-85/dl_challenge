import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitsModel(nn.Module):
    def __init__(self):
        super(DigitsModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


class CalculatorModel(nn.Module):
    def __init__(self, model_digits):
        super(CalculatorModel, self).__init__()
        
        self.model_digits = model_digits
        
        self.classifier = nn.Sequential(
          nn.Linear(24, 240),
          nn.ReLU(),
          nn.Linear(240, 96),
          nn.LogSoftmax(dim=1),
        )

    def forward(self, data_1, data_2, op):
        a = self.model_digits(data_1)
        b = self.model_digits(data_2)
        x = torch.cat((a.view(a.size(0), -1), b.view(b.size(0), -1), \
        op.view(op.size(0), -1)), dim=1)
        
        output = self.classifier(x)
        return output