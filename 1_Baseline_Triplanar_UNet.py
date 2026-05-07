!pip install segmentation_models_pytorch nibabel

import os
import re
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from scipy.ndimage import label
import warnings
warnings.filterwarnings('ignore')

print("🚀 BOOTING MASTER SCRIPT: 40-Patient Train & 10-Patient Test...")

# ==========================================
# 1. SETUP & SETTINGS
# ==========================================
base_dir = '/content/drive/MyDrive/50 patients' 
weights_path = '/content/drive/MyDrive/MASTER_BASELINE_100Epochs.pth'
num_epochs = 100 
MIN_LESION_VOXELS = 3

# ==========================================
# 2. YOUR ORIGINAL LOSS (BCE + Dice)
# ==========================================
class FastDiceBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.bce = torch.nn.BCEWithLogitsLoss() 

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.bce(y_pred, y_true)

# ==========================================
# 3. TRAINING DATASET (40 Patients Only)
# ==========================================
class MasterTrainingDataset(Dataset):
    def __init__(self, base_dir, max_patients=40):
        self.base_dir = base_dir
        all_items = sorted(os.listdir(base_dir))
        # strictly the FIRST 40 patients
        self.patient_folders = [f for f in all_items if os.path.isdir(os.path.join(base_dir, f))][:max_patients]
        self.all_slices = self._extract_all_orientations()

    def _normalize(self, image):
        mean_val, std_val = np.mean(image), np.std(image)
        return (image - mean_val) / std_val if std_val > 0 else image - mean_val

    def _extract_all_orientations(self):
        print(f"🔪 Slicing 40 training patients across all 3 planes...")
        slices_data = []
        for pid in self.patient_folders:
            p_dir = os.path.join(self.base_dir, pid)
            
            # Safe loader to prevent crashes
            try:
                t1_f = [f for f in os.listdir(p_dir) if re.match(rf"{pid}_\d{{8}}\.nii\.gz", f) and not f.startswith('._')][0]
                t2_f = [f for f in os.listdir(p_dir) if "to_FLAIR1.nii.gz" in f and not f.startswith('._')][0]
                lbl_f = [f for f in os.listdir(p_dir) if "Segmentation-Segment_1-label.nii.gz" in f and not f.startswith('._')][0]
            except IndexError:
                print(f"⚠️ Skipping Patient {pid}: Missing files.")
                continue
                
            t1 = self._normalize(nib.load(os.path.join(p_dir, t1_f)).get_fdata()).astype(np.float32)
            t2 = self._normalize(nib.load(os.path.join(p_dir, t2_f)).get_fdata()).astype(np.float32)
            lbl = nib.load(os.path.join(p_dir, lbl_f)).get_fdata().astype(np.int8)
            
            mx, my, mz = min(t1.shape[0], lbl.shape[0]), min(t1.shape[1], lbl.shape[1]), min(t1.shape[2], lbl.shape[2])
            t1, t2, lbl = t1[:mx, :my, :mz], t2[:mx, :my, :mz], lbl[:mx, :my, :mz]
            
            # Triplanar Extraction
            for i in range(mz):
                if np.sum(lbl[:, :, i]) > 0 or i % 5 == 0: slices_data.append((t1[:, :, i], t2[:, :, i], lbl[:, :, i]))
            for i in range(my):
                if np.sum(lbl[:, i, :]) > 0 or i % 5 == 0: slices_data.append((t1[:, i, :], t2[:, i, :], lbl[:, i, :]))
            for i in range(mx):
                if np.sum(lbl[i, :, :]) > 0 or i % 5 == 0: slices_data.append((t1[i, :, :], t2[i, :, :], lbl[i, :, :]))
                    
        return slices_data

    def __len__(self): return len(self.all_slices)

    def __getitem__(self, idx):
        t1, t2, lbl = self.all_slices[idx]
        img = torch.tensor(np.stack([t1, t2], axis=0), dtype=torch.float32)
        lbl = torch.tensor(lbl, dtype=torch.float32).unsqueeze(0)
        img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        lbl = F.interpolate(lbl.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
        
        # Simple data augmentation
        if torch.rand(1) > 0.5: img, lbl = torch.flip(img, dims=[2]), torch.flip(lbl, dims=[2])
        if torch.rand(1) > 0.5: img, lbl = torch.flip(img, dims=[1]), torch.flip(lbl, dims=[1])
        return img, lbl

# ==========================================
# 4. TRAINING EXECUTION
# ==========================================
dataset = MasterTrainingDataset(base_dir, max_patients=40)
train_loader = DataLoader(dataset, batch_size=20, shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=2, classes=1).to(device)

criterion = FastDiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print(f"\n🔥 IGNITION: Training on 40 patients for {num_epochs} Epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f}")
    
torch.save(model.state_dict(), weights_path)
print(f"💾 Training Complete! Weights saved to: {weights_path}")


# ==========================================
# 5. TESTING EXECUTION (The 10 Hold-out Patients)
# ==========================================
print("\n🔍 RUNNING 10-PATIENT TEST SUITE...")
model.eval()
all_items = sorted(os.listdir(base_dir))
# strictly the LAST 10 patients
test_patients = [f for f in all_items if os.path.isdir(os.path.join(base_dir, f))][40:50]

tot_gt_slices = 0
tot_det_slices = 0
all_dice = []

for pid in test_patients:
    p_dir = os.path.join(base_dir, pid)
    try:
        t1_f = [f for f in os.listdir(p_dir) if re.match(rf"{pid}_\d{{8}}\.nii\.gz", f) and not f.startswith('._')][0]
        t2_f = [f for f in os.listdir(p_dir) if "to_FLAIR1.nii.gz" in f and not f.startswith('._')][0]
        lbl_f = [f for f in os.listdir(p_dir) if "Segmentation-Segment_1-label.nii.gz" in f and not f.startswith('._')][0]
    except: continue

    t1 = dataset._normalize(nib.load(os.path.join(p_dir, t1_f)).get_fdata()).astype(np.float32)
    t2 = dataset._normalize(nib.load(os.path.join(p_dir, t2_f)).get_fdata()).astype(np.float32)
    gt = nib.load(os.path.join(p_dir, lbl_f)).get_fdata().astype(np.int8)
    
    mx, my, mz = min(t1.shape[0], gt.shape[0]), min(t1.shape[1], gt.shape[1]), min(t1.shape[2], gt.shape[2])
    t1, t2, gt = t1[:mx, :my, :mz], t2[:mx, :my, :mz], gt[:mx, :my, :mz]

    p_ax, p_cor, p_sag = np.zeros((mx, my, mz)), np.zeros((mx, my, mz)), np.zeros((mx, my, mz))

    # Slice-by-slice 3D reconstruction
    with torch.no_grad():
        for i in range(mz):
            img = torch.tensor(np.stack([t1[:,:,i], t2[:,:,i]], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            p_ax[:,:,i] = F.interpolate(torch.sigmoid(model(F.interpolate(img, size=(256,256)))), size=(mx,my)).squeeze().cpu().numpy()
        for i in range(my):
            img = torch.tensor(np.stack([t1[:,i,:], t2[:,i,:]], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            p_cor[:,i,:] = F.interpolate(torch.sigmoid(model(F.interpolate(img, size=(256,256)))), size=(mx,mz)).squeeze().cpu().numpy()
        for i in range(mx):
            img = torch.tensor(np.stack([t1[i,:,:], t2[i,:,:]], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            p_sag[i,:,:] = F.interpolate(torch.sigmoid(model(F.interpolate(img, size=(256,256)))), size=(my,mz)).squeeze().cpu().numpy()

    # Averaging & Thresholding
    fused = (p_ax + p_cor + p_sag) / 3.0
    raw_pred = (fused >= 0.50).astype(np.float32)
    
    # Pruning tiny dust particles
    labeled_array, num_features = label(raw_pred)
    final_pred = np.zeros_like(raw_pred)
    for blob_id in range(1, num_features + 1):
        blob_mask = (labeled_array == blob_id)
        if np.sum(blob_mask) >= MIN_LESION_VOXELS:
            final_pred[blob_mask] = 1.0
            
    # Calculate Dice Score
    inter = (final_pred * gt).sum()
    patient_dice = ((2.0 * inter) / (final_pred.sum() + gt.sum() + 1e-6)) * 100
    all_dice.append(patient_dice)

    # Slice Detection Logic
    slices_w_gt = [i for i in range(mz) if np.sum(gt[:,:,i]) > 0]
    det = sum([1 for i in slices_w_gt if np.sum(final_pred[:,:,i]) > 0])
    tot_gt_slices += len(slices_w_gt)
    tot_det_slices += det

    print(f"🏥 Patient {pid} | Dice: {patient_dice:5.2f}% | Slices Detected: {det}/{len(slices_w_gt)}")

print("\n==================================================")
print(f"🏆 FINAL OVERALL MEAN DICE (10 Patients): {np.mean(all_dice):.2f}%")
if tot_gt_slices > 0:
    print(f"🌍 TOTAL SLICE DETECTION RATE: {(tot_det_slices/tot_gt_slices)*100:.2f}%")
print("==================================================")
