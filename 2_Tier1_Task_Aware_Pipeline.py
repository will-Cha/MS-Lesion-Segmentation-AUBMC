!pip install segmentation_models_pytorch nibabel SimpleITK

import os
import re
import random
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from scipy.ndimage import label
import warnings
warnings.filterwarnings('ignore')

print("🚀 BOOTING TIER 1 TASK-AWARE ARCHITECTURE (Difference Channel + HistMatch + TTA)...")

# ==========================================
# 1. SETUP & SETTINGS
# ==========================================
base_dir = '/content/drive/MyDrive/50 patients' 
weights_path = '/content/drive/MyDrive/TIER_1_TASK_AWARE_100Epochs.pth'
num_epochs = 100 
MIN_LESION_VOXELS = 3

# ==========================================
# 2. CLAUDE'S CORE FIXES: HistMatch & Tversky
# ==========================================
def hist_match(moving, fixed):
    """Matches the histogram of FLAIR2 (moving) to FLAIR1 (fixed)"""
    f = sitk.GetImageFromArray(fixed.astype(np.float32))
    m = sitk.GetImageFromArray(moving.astype(np.float32))
    matched = sitk.HistogramMatching(m, f, numberOfHistogramLevels=256, numberOfMatchPoints=10)
    return sitk.GetArrayFromImage(matched)

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.a = alpha # Weight for False Negatives
        self.b = beta  # Weight for False Positives
        self.g = gamma

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        TP = (p * target).sum()
        FN = ((1 - p) * target).sum()
        FP = (p * (1 - target)).sum()
        T = (TP + 1e-6) / (TP + self.a * FN + self.b * FP + 1e-6)
        return (1 - T) ** self.g

# ==========================================
# 3. BALANCED TRIPLANAR DATASET (3-Channel: T1, T2, Diff)
# ==========================================
class TaskAwareDataset(Dataset):
    def __init__(self, base_dir, max_patients=40):
        self.base_dir = base_dir
        all_items = sorted(os.listdir(base_dir))
        self.patient_folders = [f for f in all_items if os.path.isdir(os.path.join(base_dir, f))][:max_patients]
        self.all_slices = self._extract_balanced()

    def _normalize(self, image):
        m, s = np.mean(image), np.std(image)
        return (image - m) / s if s > 0 else image - m

    def _extract_balanced(self):
        print(f"🔪 Slicing 40 patients with Histogram Matching & 50/50 Foreground/Background Balance...")
        pos_slices = []
        neg_slices = []
        
        for pid in self.patient_folders:
            p_dir = os.path.join(self.base_dir, pid)
            try:
                t1_f = [f for f in os.listdir(p_dir) if re.match(rf"{pid}_\d{{8}}\.nii\.gz", f) and not f.startswith('._')][0]
                t2_f = [f for f in os.listdir(p_dir) if "to_FLAIR1.nii.gz" in f and not f.startswith('._')][0]
                lbl_f = [f for f in os.listdir(p_dir) if "Segmentation-Segment_1-label.nii.gz" in f and not f.startswith('._')][0]
            except IndexError: continue
                
            t1_raw = nib.load(os.path.join(p_dir, t1_f)).get_fdata().astype(np.float32)
            t2_raw = nib.load(os.path.join(p_dir, t2_f)).get_fdata().astype(np.float32)
            lbl = nib.load(os.path.join(p_dir, lbl_f)).get_fdata().astype(np.int8)
            
            # 🛡️ THE FIX: Crop everything to the same size FIRST before doing math!
            mx = min(t1_raw.shape[0], t2_raw.shape[0], lbl.shape[0])
            my = min(t1_raw.shape[1], t2_raw.shape[1], lbl.shape[1])
            mz = min(t1_raw.shape[2], t2_raw.shape[2], lbl.shape[2])
            
            t1_raw = t1_raw[:mx, :my, :mz]
            t2_raw = t2_raw[:mx, :my, :mz]
            lbl = lbl[:mx, :my, :mz]
            
            # STEP 1: Histogram Match T2 to T1
            t2_matched = hist_match(t2_raw, t1_raw)
            
            # STEP 2: Normalize
            t1 = self._normalize(t1_raw)
            t2 = self._normalize(t2_matched)
            
            # STEP 3: Compute Difference Map safely
            diff = t2 - t1
            
            # Separate positive and negative slices
            for planes in [(t1, t2, diff, lbl, mz, 2), (t1, t2, diff, lbl, my, 1), (t1, t2, diff, lbl, mx, 0)]:
                t1_p, t2_p, diff_p, lbl_p, limit, axis = planes
                for i in range(limit):
                    if axis == 2:
                        s_t1, s_t2, s_diff, s_lbl = t1_p[:,:,i], t2_p[:,:,i], diff_p[:,:,i], lbl_p[:,:,i]
                    elif axis == 1:
                        s_t1, s_t2, s_diff, s_lbl = t1_p[:,i,:], t2_p[:,i,:], diff_p[:,i,:], lbl_p[:,i,:]
                    else:
                        s_t1, s_t2, s_diff, s_lbl = t1_p[i,:,:], t2_p[i,:,:], diff_p[i,:,:], lbl_p[i,:,:]
                        
                    if np.sum(s_lbl) > 0:
                        pos_slices.append((s_t1, s_t2, s_diff, s_lbl))
                    else:
                        neg_slices.append((s_t1, s_t2, s_diff, s_lbl))
                        
        # BALANCE THE DATASET (50% Lesion, 50% Empty)
        random.shuffle(neg_slices)
        balanced_neg = neg_slices[:len(pos_slices)]
        final_dataset = pos_slices + balanced_neg
        random.shuffle(final_dataset)
        
        print(f"⚖️ Dataset Balanced: {len(pos_slices)} Positive Slices, {len(balanced_neg)} Negative Slices.")
        return final_dataset

    def __len__(self): return len(self.all_slices)

    def __getitem__(self, idx):
        t1, t2, diff, lbl = self.all_slices[idx]
        img = torch.tensor(np.stack([t1, t2, diff], axis=0), dtype=torch.float32)
        lbl = torch.tensor(lbl, dtype=torch.float32).unsqueeze(0)
        
        img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        lbl = F.interpolate(lbl.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
        
        if torch.rand(1) > 0.5: img, lbl = torch.flip(img, dims=[2]), torch.flip(lbl, dims=[2])
        if torch.rand(1) > 0.5: img, lbl = torch.flip(img, dims=[1]), torch.flip(lbl, dims=[1])
        return img, lbl

# ==========================================
# 4. TRAINING EXECUTION
# ==========================================
dataset = TaskAwareDataset(base_dir, max_patients=40)
train_loader = DataLoader(dataset, batch_size=24, shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Back to ResNet34, but with 3 Input Channels!
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)

criterion = FocalTverskyLoss(alpha=0.7, beta=0.3)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

print(f"\n🔥 IGNITION: Training Tier 1 Task-Aware Model for {num_epochs} Epochs...")

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
        
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}] | Tversky Loss: {running_loss / len(train_loader):.4f}")
    
torch.save(model.state_dict(), weights_path)
print(f"💾 Training Complete! Weights saved to: {weights_path}")

# ==========================================
# 5. INFERENCE WITH TTA (Test-Time Augmentation)
# ==========================================
print("\n🔍 RUNNING 10-PATIENT TEST SUITE WITH TTA...")
model.eval()
all_items = sorted(os.listdir(base_dir))
test_patients = [f for f in all_items if os.path.isdir(os.path.join(base_dir, f))][40:50]

def predict_tta(model, img_tensor):
    p_orig = torch.sigmoid(model(img_tensor))
    p_hflip = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [3]))), [3])
    p_vflip = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [2]))), [2])
    p_both = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [2, 3]))), [2, 3])
    return (p_orig + p_hflip + p_vflip + p_both) / 4.0

tot_gt_slices, tot_det_slices = 0, 0
all_dice = []

for pid in test_patients:
    p_dir = os.path.join(base_dir, pid)
    try:
        t1_f = [f for f in os.listdir(p_dir) if re.match(rf"{pid}_\d{{8}}\.nii\.gz", f) and not f.startswith('._')][0]
        t2_f = [f for f in os.listdir(p_dir) if "to_FLAIR1.nii.gz" in f and not f.startswith('._')][0]
        lbl_f = [f for f in os.listdir(p_dir) if "Segmentation-Segment_1-label.nii.gz" in f and not f.startswith('._')][0]
    except: continue

    t1_raw = nib.load(os.path.join(p_dir, t1_f)).get_fdata().astype(np.float32)
    t2_raw = nib.load(os.path.join(p_dir, t2_f)).get_fdata().astype(np.float32)
    gt = nib.load(os.path.join(p_dir, lbl_f)).get_fdata().astype(np.int8)
    
    # 🛡️ THE FIX: Crop in Inference too!
    mx = min(t1_raw.shape[0], t2_raw.shape[0], gt.shape[0])
    my = min(t1_raw.shape[1], t2_raw.shape[1], gt.shape[1])
    mz = min(t1_raw.shape[2], t2_raw.shape[2], gt.shape[2])
    
    t1_raw, t2_raw, gt = t1_raw[:mx, :my, :mz], t2_raw[:mx, :my, :mz], gt[:mx, :my, :mz]
    
    # Preprocess
    t2_matched = hist_match(t2_raw, t1_raw)
    t1 = dataset._normalize(t1_raw)
    t2 = dataset._normalize(t2_matched)
    diff = t2 - t1

    p_ax, p_cor, p_sag = np.zeros((mx, my, mz)), np.zeros((mx, my, mz)), np.zeros((mx, my, mz))

    with torch.no_grad():
        for i in range(mz):
            img = torch.tensor(np.stack([t1[:,:,i], t2[:,:,i], diff[:,:,i]], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            p_ax[:,:,i] = F.interpolate(predict_tta(model, F.interpolate(img, size=(256,256))), size=(mx,my)).squeeze().cpu().numpy()
        for i in range(my):
            img = torch.tensor(np.stack([t1[:,i,:], t2[:,i,:], diff[:,i,:]], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            p_cor[:,i,:] = F.interpolate(predict_tta(model, F.interpolate(img, size=(256,256))), size=(mx,mz)).squeeze().cpu().numpy()
        for i in range(mx):
            img = torch.tensor(np.stack([t1[i,:,:], t2[i,:,:], diff[i,:,:]], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            p_sag[i,:,:] = F.interpolate(predict_tta(model, F.interpolate(img, size=(256,256))), size=(my,mz)).squeeze().cpu().numpy()

    fused = (p_ax + p_cor + p_sag) / 3.0
    raw_pred = (fused >= 0.50).astype(np.float32)
    
    labeled_array, num_features = label(raw_pred)
    final_pred = np.zeros_like(raw_pred)
    for blob_id in range(1, num_features + 1):
        blob_mask = (labeled_array == blob_id)
        if np.sum(blob_mask) >= MIN_LESION_VOXELS:
            final_pred[blob_mask] = 1.0
            
    inter = (final_pred * gt).sum()
    patient_dice = ((2.0 * inter) / (final_pred.sum() + gt.sum() + 1e-6)) * 100
    all_dice.append(patient_dice)

    slices_w_gt = [i for i in range(mz) if np.sum(gt[:,:,i]) > 0]
    det = sum([1 for i in slices_w_gt if np.sum(final_pred[:,:,i]) > 0])
    tot_gt_slices += len(slices_w_gt)
    tot_det_slices += det

    print(f"🏥 Patient {pid} | Dice: {patient_dice:5.2f}% | Slices Detected: {det}/{len(slices_w_gt)}")

print("\n==================================================")
print(f"🏆 TIER 1 TASK-AWARE FINAL MEAN DICE: {np.mean(all_dice):.2f}%")
if tot_gt_slices > 0:
    print(f"🌍 TIER 1 SLICE DETECTION RATE: {(tot_det_slices/tot_gt_slices)*100:.2f}%")
print("==================================================")
