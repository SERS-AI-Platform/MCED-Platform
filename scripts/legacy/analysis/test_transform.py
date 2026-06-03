import sys, importlib
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import logging; logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import src.sers.io; importlib.reload(src.sers.io)
from src.sers.io import load_dataset
from src.sers.config import load_config

config = load_config('config/config.yaml')
result = load_dataset(Path('data/equipment_test_data'),
    equipment_mapping=config.equipment_folder_to_group, pattern='*.*', show_progress=False)
fp = np.linspace(400, 2200, 900)

def snv(y):
    return (y - y.mean()) / (y.std() + 1e-10)

ps = {}
for eq in ['handheld', 'thermo', 'nanoscope', 'medical_raw']:
    sps = result.get_equipment(eq)
    ss = defaultdict(list)
    for (_, g, s, r), (x, y) in sps.items():
        try:
            yr = interp1d(x, y, bounds_error=False, fill_value=np.nan)(fp)
            n = np.isnan(yr)
            if n.all(): continue
            if n.any(): yr[n] = np.interp(fp[n], fp[~n], yr[~n])
            ss[s].append(yr)
        except:
            pass
    ps[eq] = {s: np.mean(v, axis=0) for s, v in ss.items() if len(v) >= 2}

print(f"Loaded: HH={len(ps['handheld'])}, TH={len(ps['thermo'])}, NS={len(ps['nanoscope'])}, RC={len(ps['medical_raw'])}")

class TF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(900, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 900))
        self.a = nn.Parameter(torch.tensor(0.3))
    def forward(self, x):
        return self.a * self.fc(x) + (1 - self.a) * x

NM = {'thermo': 'Thermo DXR3xi', 'nanoscope': 'NS200', 'medical_raw': 'RamCheck-A1'}

for seq in ['thermo', 'nanoscope', 'medical_raw']:
    com = sorted(set(ps['handheld']) & set(ps[seq]))
    tr, te = train_test_split(com, test_size=0.2, random_state=42)

    trs = torch.FloatTensor(np.array([snv(ps[seq][s]) for s in tr]))
    trt = torch.FloatTensor(np.array([snv(ps['handheld'][s]) for s in tr]))
    tes = torch.FloatTensor(np.array([snv(ps[seq][s]) for s in te]))
    tet = np.array([snv(ps['handheld'][s]) for s in te])

    bef = [float(np.corrcoef(tes[i].numpy(), tet[i])[0, 1]) for i in range(len(tes))]

    m = TF()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(100):
        m.train()
        p = m(trs)
        mse = nn.functional.mse_loss(p, trt)
        pc = p - p.mean(1, keepdim=True)
        tc = trt - trt.mean(1, keepdim=True)
        cl = (1 - (pc * tc).sum(1) / (torch.sqrt((pc**2).sum(1) * (tc**2).sum(1)) + 1e-10)).mean()
        loss = mse + cl
        opt.zero_grad()
        loss.backward()
        opt.step()

    m.eval()
    with torch.no_grad():
        pred = m(tes).numpy()
    aft = [float(np.corrcoef(pred[i], tet[i])[0, 1]) for i in range(len(pred))]
    print(f"{NM[seq]:>12s}: before={np.mean(bef):.4f}  after={np.mean(aft):.4f}  delta={np.mean(aft)-np.mean(bef):+.4f}")

print("Done.")
