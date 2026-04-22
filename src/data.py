import os
import random
from os.path import join

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
import pydicom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def bit_get(val, idx):
    return (val >> idx) & 1


def create_pascal_label_colormap():
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3
    return colormap


def create_chaos_colormap():
    """
    0 = background  (black)
    1 = liver        (red)
    2 = right kidney (dark blue)
    3 = left kidney  (light blue)
    4 = spleen       (yellow)
    """
    return np.array([
        [0,   0,   0],
        [255,  0,   0],
        [0,   0, 255],
        [0, 128, 255],
        [255, 255,  0],
    ], dtype=np.uint8)


# ---------------------------------------------------------------------------
# CHAOS label pixel → class index mappings
#
# CT  Ground PNGs:  0 = background, 255 = liver
# MRI Ground PNGs:  0 = background, 63 = liver, 126 = right kidney,
#                   189 = left kidney, 252 = spleen
# ---------------------------------------------------------------------------

_CT_PIXEL_TO_CLASS  = {0: 0, 255: 1}
_MRI_PIXEL_TO_CLASS = {0: 0, 63: 1, 126: 2, 189: 3, 252: 4}


# ---------------------------------------------------------------------------
# CHAOS Dataset
# ---------------------------------------------------------------------------
#
# Exact on-disk layout (verified from screenshots):
#
#   <pytorch_data_dir>/
#     archive/
#       CHAOS_Train_Sets/
#         Train_Sets/
#           CT/
#             1/
#               DICOM_anon/   *.dcm
#               Ground/       liver_GT_000.png, liver_GT_001.png ...
#             2/ 5/ 6/ 8/ ...
#           MR/
#             <pid>/
#               T1DUAL/
#                 DICOM_anon/
#                   InPhase/  *.dcm
#                 Ground/     liver_GT_000.png ...
#               T2SPIR/
#                 DICOM_anon/ *.dcm
#                 Ground/     liver_GT_000.png ...
# ---------------------------------------------------------------------------

class CHAOS(Dataset):
    """
    Parameters
    ----------
    root             : str  – pytorch_data_dir (contains 'archive/')
    modality         : str  – 'CT' | 'T1DUAL' | 'T2SPIR' | 'all'
    image_set        : str  – 'train' | 'val' | 'all'
    transform        : PIL Image -> tensor [3, H, W]
    target_transform : PIL Image -> tensor [1, H, W]  (nearest, no interp)
    n_classes        : int – 2 (CT liver only) or 5 (all MRI organs)
    """

    MODALITIES = ("CT", "T1DUAL", "T2SPIR")

    def __init__(self, root, modality, image_set, transform, target_transform,
                 n_classes=5):
        super().__init__()
        assert modality in (*self.MODALITIES, "all"), \
            f"modality must be one of {self.MODALITIES + ('all',)}"
        assert image_set in ("train", "val", "all")

        self.modality         = modality
        self.image_set        = image_set
        self.transform        = transform
        self.target_transform = target_transform
        self.n_classes        = n_classes

        # Root of the actual CHAOS train data (verified path)
        self.train_root = join(root, "archive", "CHAOS_Train_Sets", "Train_Sets")

        # Each entry: (dcm_path, mask_path_or_None, modality_tag, patient_id)
        self.samples: list = []
        self._collect_samples()

        # Patient-level 80/20 split — all slices of a patient stay together
        # so there is no cross-patient leakage between train and val.
        if image_set != "all":
            all_patients = sorted(set(s[3] for s in self.samples))
            n_val = max(1, int(0.2 * len(all_patients)))
            val_patients = set(all_patients[-n_val:])
            if image_set == "val":
                self.samples = [s for s in self.samples if s[3] in val_patients]
            else:
                self.samples = [s for s in self.samples if s[3] not in val_patients]

    # ------------------------------------------------------------------
    # Sample collection — exact paths from screenshots
    # ------------------------------------------------------------------

    def _collect_samples(self):
        mods_wanted = self.MODALITIES if self.modality == "all" \
                      else (self.modality,)

        if "CT" in mods_wanted:
            ct_root = join(self.train_root, "CT")
            if os.path.isdir(ct_root):
                for pid in sorted(os.listdir(ct_root),
                                  key=lambda x: int(x) if x.isdigit() else 0):
                    self._add_ct(join(ct_root, pid), pid)

        for seq in ("T1DUAL", "T2SPIR"):
            if seq in mods_wanted:
                mr_root = join(self.train_root, "MR")
                if os.path.isdir(mr_root):
                    for pid in sorted(os.listdir(mr_root),
                                      key=lambda x: int(x) if x.isdigit() else 0):
                        self._add_mri(join(mr_root, pid), pid, seq)

    def _add_ct(self, patient_path: str, patient_id: str):
        """
        CT/<pid>/DICOM_anon/*.dcm  paired with  CT/<pid>/Ground/liver_GT_NNN.png
        Ground/ contains PNGs directly — no nested subfolder.
        """
        dicom_dir  = join(patient_path, "DICOM_anon")
        ground_dir = join(patient_path, "Ground")

        if not os.path.isdir(dicom_dir):
            return

        dcm_files  = sorted(f for f in os.listdir(dicom_dir)
                             if f.lower().endswith(".dcm"))
        mask_files = sorted(f for f in os.listdir(ground_dir)
                             if f.lower().endswith(".png")) \
                     if os.path.isdir(ground_dir) else []

        for i, dcm in enumerate(dcm_files):
            mask_path = join(ground_dir, mask_files[i]) \
                        if i < len(mask_files) else None
            self.samples.append((join(dicom_dir, dcm), mask_path, "CT", patient_id))

    def _add_mri(self, patient_path: str, patient_id: str, seq: str):
        """
        MR/<pid>/T1DUAL/DICOM_anon/InPhase/*.dcm  ↔  MR/<pid>/T1DUAL/Ground/*.png
        MR/<pid>/T2SPIR/DICOM_anon/*.dcm           ↔  MR/<pid>/T2SPIR/Ground/*.png
        Ground/ contains PNGs directly — no nested patient subfolder.
        """
        seq_path = join(patient_path, seq)
        if not os.path.isdir(seq_path):
            return

        dicom_dir = join(seq_path, "DICOM_anon", "InPhase") \
                    if seq == "T1DUAL" else join(seq_path, "DICOM_anon")
        ground_dir = join(seq_path, "Ground")

        if not os.path.isdir(dicom_dir):
            return

        dcm_files  = sorted(f for f in os.listdir(dicom_dir)
                             if f.lower().endswith(".dcm"))
        mask_files = sorted(f for f in os.listdir(ground_dir)
                             if f.lower().endswith(".png")) \
                     if os.path.isdir(ground_dir) else []

        for i, dcm in enumerate(dcm_files):
            mask_path = join(ground_dir, mask_files[i]) \
                        if i < len(mask_files) else None
            self.samples.append((join(dicom_dir, dcm), mask_path, seq, patient_id))

    # ------------------------------------------------------------------
    # DICOM → 8-bit RGB PIL Image
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dicom_as_pil(dcm_path: str) -> Image.Image:
        dcm = pydicom.dcmread(dcm_path)
        arr = dcm.pixel_array.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    # ------------------------------------------------------------------
    # Ground PNG → uint8 class-index PIL Image (mode "L")
    # 255 is used as the ignore sentinel in uint8 space;
    # it is remapped to -1 (int64) inside __getitem__.
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mask(mask_path: str, modality: str) -> Image.Image:
        raw = np.array(Image.open(mask_path).convert("L"))   # uint8 grayscale
        lut = _CT_PIXEL_TO_CLASS if modality == "CT" else _MRI_PIXEL_TO_CLASS
        label = np.full(raw.shape, 255, dtype=np.uint8)      # 255 = ignore
        for px, cls in lut.items():
            label[raw == px] = cls
        return Image.fromarray(label, mode="L")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        dcm_path, mask_path, mod, _ = self.samples[index]

        img = self._load_dicom_as_pil(dcm_path)

        seed = np.random.randint(2147483647)

        random.seed(seed); torch.manual_seed(seed)
        img = self.transform(img)                            # [3, H, W]

        if mask_path is not None and os.path.exists(mask_path):
            mask_pil = self._load_mask(mask_path, mod)
            random.seed(seed); torch.manual_seed(seed)
            label = self.target_transform(mask_pil).squeeze(0).long()  # [H, W]
            label[label == 255] = -1                         # restore ignore sentinel
        else:
            label = torch.full(img.shape[1:], -1, dtype=torch.long)

        valid_mask = (label >= 0).float()
        return img, label, valid_mask


# ---------------------------------------------------------------------------
# MaterializedDataset
# ---------------------------------------------------------------------------

class MaterializedDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.materialized = []
        loader = DataLoader(ds, num_workers=4, collate_fn=lambda l: l[0])
        for batch in tqdm(loader):
            self.materialized.append(batch)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        return self.materialized[ind]


# ---------------------------------------------------------------------------
# ContrastiveSegDataset — EAGLE-compatible wrapper
# ---------------------------------------------------------------------------
#
# In your Hydra / train config set:
#   dataset_name    : chaos
#   chaos_modality  : CT | T1DUAL | T2SPIR | all   (default: all)
#   chaos_n_classes : 5                             (default: 5)
#
# pytorch_data_dir must point to the folder that contains "archive/", i.e.:
#   /content/drive/MyDrive/EAGLE/src_EAGLE/pytorch_data_dir
# ---------------------------------------------------------------------------

class ContrastiveSegDataset(Dataset):

    def __init__(
        self,
        pytorch_data_dir,
        dataset_name,
        crop_type,              # ignored — kept for API compatibility
        image_set,
        transform,
        target_transform,
        cfg,
        aug_geometric_transform=None,
        aug_photometric_transform=None,
        mask=False,
        extra_transform=None,
        model_type_override=None,
    ):
        super().__init__()

        # assert dataset_name == "chaos", \
        #     f"This datasets.py only supports dataset_name='chaos', got '{dataset_name}'"

        self.mask                      = mask
        self.extra_transform           = extra_transform
        self.aug_photometric_transform = aug_photometric_transform

        modality  = getattr(cfg, "chaos_modality",  "all")
        n_classes = getattr(cfg, "chaos_n_classes", 5)
        self.n_classes = n_classes

        self.dataset = CHAOS(
            root             = pytorch_data_dir,
            modality         = modality,
            image_set        = image_set,
            transform        = transform,
            target_transform = target_transform,
            n_classes        = n_classes,
        )

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, ind):
        pack = self.dataset[ind]   # (img, label, valid_mask)

        seed = np.random.randint(2147483647)
        self._set_seed(seed)

        # Spatial coordinate grid — used by some EAGLE loss terms
        coord_entries = torch.meshgrid(
            [torch.linspace(-1, 1, pack[0].shape[1]),
             torch.linspace(-1, 1, pack[0].shape[2])],
            indexing="ij"
        )
        coord = torch.cat([t.unsqueeze(0) for t in coord_entries], 0)

        extra = self.extra_transform \
                if self.extra_transform is not None else lambda i, x: x

        ret = {
            "ind":       ind,
            "img":       extra(ind, pack[0]),
            "label":     extra(ind, pack[1]),
            "img_pos":   extra(ind, pack[0]),
            "label_pos": extra(ind, pack[1]),
            "coord":     coord,
        }

        if self.mask:
            ret["mask"] = pack[2]

        if self.aug_photometric_transform is not None:
            ret["img_pos_aug"] = self.aug_photometric_transform(ret["img_pos"])

        return ret