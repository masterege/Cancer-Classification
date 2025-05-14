import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import json
import os

class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)  # long for CrossEntropy

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CancerNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CancerNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_nn_classifier(X, y, num_epochs=25, batch_size=64, lr=0.001, output_dir="outputs/nn"):
    os.makedirs(output_dir, exist_ok=True)

    # Encode target labels to integers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Dataset & DataLoader
    train_data = CancerDataset(X_train, y_train)
    test_data = CancerDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = CancerNet(input_dim=X.shape[1], output_dim=len(np.unique(y_encoded)))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    y_preds, y_true = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            output = model(xb)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            y_preds.extend(preds)
            y_true.extend(yb.numpy())

    acc = accuracy_score(y_true, y_preds)
    report = classification_report(y_true, y_preds, output_dict=True)
    print(f"\n[INFO] Test Accuracy: {acc:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "cancer_nn.pth"))
    print(f"[INFO] Saved model to {output_dir}/cancer_nn.pth")

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=4)
    print(f"[INFO] Saved classification report to {output_dir}/metrics.json")

    return model, report, encoder
