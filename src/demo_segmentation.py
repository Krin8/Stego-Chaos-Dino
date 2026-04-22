from modules import *
from data import *   # <-- IMPORTANT (use your CHAOS dataset)
import hydra
import torch.multiprocessing
from PIL import Image
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
import torch.nn.functional as F
import os
from os.path import join

torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.serialization

torch.serialization.add_safe_globals([ModelCheckpoint])

from pytorch_lightning.callbacks import ModelCheckpoint
from train_segmentation import LitUnsupervisedSegmenter
import torch.serialization

torch.serialization.add_safe_globals([
    ModelCheckpoint,
    LitUnsupervisedSegmenter
])

@hydra.main(config_path="configs", config_name="demo_config.yml")
def my_app(cfg: DictConfig) -> None:
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "linear"), exist_ok=True)
    os.makedirs(join(result_dir, "img"), exist_ok=True)

    # Load model
    model = LitUnsupervisedSegmenter.load_from_checkpoint(
        cfg.model_path,
        weights_only=False   # ⚠️ important
    )    
    print(OmegaConf.to_yaml(model.cfg))

    # CHAOS Dataset (instead of UnlabeledImageFolder)
    dataset = ContrastiveSegDataset(
        pytorch_data_dir=cfg.pytorch_data_dir,
        dataset_name="chaos",
        crop_type=None,
        image_set="val",   # or "all"
        transform=get_transform(cfg.res, False, "center"),
        target_transform=get_transform(cfg.res, True, "center"),
        cfg=model.cfg,
        mask=True
    )

    loader = DataLoader(
        dataset,
        cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate
    )
    print("Dataset size:", len(dataset))
    model.eval().cuda()

    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net

    label_cmap = model.label_cmap

    for i, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            img = batch["img"].cuda()
            label = batch["label"]   # not used, but useful if needed

            # ---- SAME LOGIC (unchanged) ----
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            # IMPORTANT FIX: use softmax (CRF expects probabilities)
            linear_probs = torch.softmax(model.linear_probe(code), dim=1).cpu()
            cluster_loss, cluster_probs = model.cluster_probe(code, None)
            cluster_probs = torch.softmax(cluster_probs, dim=1).cpu()

            # --------------------------------

            for j in range(img.shape[0]):
                single_img = img[j].cpu()

                linear_crf = dense_crf(single_img, linear_probs[j]).argmax(0)
                cluster_crf = dense_crf(single_img, cluster_probs[j]).argmax(0)

                idx = i * loader.batch_size + j

                # Save original image
                plot_img = (prep_for_plot(single_img) * 255).numpy().astype(np.uint8)
                Image.fromarray(plot_img).save(join(result_dir, "img", f"{idx}.jpg"))

                # Save linear prediction (colored)
                linear_color = label_cmap[linear_crf]
                Image.fromarray(linear_color.astype(np.uint8)).save(
                    join(result_dir, "linear", f"{idx}.png")
                )

                # Save cluster prediction (mapped + colored)
                # mapped_cluster = model.test_cluster_metrics.map_clusters(
                #     torch.from_numpy(cluster_crf)
                # )

                cluster_color = label_cmap[cluster_crf]
                Image.fromarray(cluster_color.astype(np.uint8)).save(
                    join(result_dir, "cluster", f"{idx}.png")
                )


if __name__ == "__main__":
    prep_args()
    my_app()