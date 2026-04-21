import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out)

def train_lstm(X, y, prediction_type, device=device):

    X = torch.tensor(X, dtype=torch.float32)   
    y = torch.tensor(y, dtype=torch.float32)

    model = LSTMModel(
        input_size=X.shape[2],
        hidden_size=64,   #  reduce from 128
        output_size=y.shape[1] if len(y.shape) > 1 else 1
    ).to(device)

    if prediction_type == "fault":
        pos_weight = (len(y) - y.sum()) / (y.sum() + 1e-6)
        pos_weight = pos_weight.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    batch_size = 32   #  IMPORTANT

    for epoch in range(100):
        model.train()

        for i in range(0, len(X), batch_size):

            X_batch = X[i:i+batch_size].to(device)
            y_batch = y[i:i+batch_size].to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            if prediction_type == "fault":
                loss = criterion(outputs.squeeze(), y_batch)
            else:
                loss = criterion(outputs, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        torch.cuda.empty_cache()  #  helps fragmentation

    return model