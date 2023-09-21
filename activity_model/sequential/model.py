import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, feature_vector_size,
                 num_feature_vectors,
                 hidden_size,
                 num_classes,
                 dropout_rate=0.1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(feature_vector_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size/2), num_classes)
    
    def forward(self, x):
        lstm_output, _ = self.lstm(x) # shape: [batch_size, seq_length, hidden_size]
        last_step = lstm_output[:, -1, :]

        # out = self.fc(flat_features)
        out = self.fc1(last_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # out shape: [batch_size, num_classes]
        return out