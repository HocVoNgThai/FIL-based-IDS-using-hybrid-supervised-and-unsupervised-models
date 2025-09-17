import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

# ===== 1. DỮ LIỆU GIẢ =====
X, _ = make_classification(
    n_samples=1500, n_features=20,
    n_informative=15, n_redundant=5,
    random_state=42
)
# 90% coi là "normal", 10% "abnormal" để test sau
X = StandardScaler().fit_transform(X)
train_X = torch.tensor(X[:1200], dtype=torch.float32)  # chỉ train bằng dữ liệu bình thường
test_X  = torch.tensor(X, dtype=torch.float32)

# ===== 2. BASIC AUTO-ENCODER (bAE) – Giảm chiều =====
class BasicAE(nn.Module):
    def __init__(self, in_dim=20, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 12), nn.ReLU(),
            nn.Linear(12, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 12), nn.ReLU(),
            nn.Linear(12, in_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

bAE = BasicAE()
opt = optim.Adam(bAE.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(50):
    opt.zero_grad()
    recon, _ = bAE(train_X)
    loss = loss_fn(recon, train_X)
    loss.backward()
    opt.step()
    if (epoch+1) % 10 == 0:
        print(f"[bAE] Epoch {epoch+1}/50, Loss {loss.item():.4f}")

# Lấy đặc trưng giảm chiều
with torch.no_grad():
    z_train = bAE.encoder(train_X).numpy()
    z_all   = bAE.encoder(test_X).numpy()

# ===== 3. ONE-CLASS SVM – Phát hiện bất thường =====
ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(z_train)
svm_preds = ocsvm.predict(z_all)   # 1 = normal, -1 = anomaly
print("Anomalies detected:", np.sum(svm_preds == -1))

# Chia dữ liệu abnormal cho bước kế tiếp
abnormal_idx = np.where(svm_preds == -1)[0]
abnormal_data = test_X[abnormal_idx]

# ===== 4. DEEP AUTO-ENCODER (dAE) – Xử lý dữ liệu bất thường =====
class DeepAE(nn.Module):
    def __init__(self, in_dim=20, latent=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, in_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

if len(abnormal_data) > 0:
    dAE = DeepAE()
    opt2 = optim.Adam(dAE.parameters(), lr=1e-3)
    for epoch in range(30):
        opt2.zero_grad()
        recon, _ = dAE(abnormal_data)
        loss = loss_fn(recon, abnormal_data)
        loss.backward()
        opt2.step()
        if (epoch+1) % 10 == 0:
            print(f"[dAE] Epoch {epoch+1}/30, Loss {loss.item():.4f}")

    with torch.no_grad():
        z_abnormal = dAE.encoder(abnormal_data)
    print("Deep latent vectors shape:", z_abnormal.shape)
else:
    print("Không phát hiện điểm bất thường nào.")
