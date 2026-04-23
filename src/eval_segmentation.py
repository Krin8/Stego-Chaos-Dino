from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import seaborn as sns
import torch.multiprocessing
try:
    from crf import dense_crf
except ImportError:
    dense_crf = None
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels

import os
from os.path import join

import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

torch.serialization.add_safe_globals([ModelCheckpoint])

def plot_cm(histogram, label_cmap, cfg):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    hist = histogram.detach().cpu().to(torch.float32)
    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues", cbar=False)
    ax.set_title('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)

    names = get_class_labels(cfg.dataset_name)
    if cfg.extra_clusters:
        names = names + ["Extra"]

    ax.set_xticks(np.arange(0, len(names)) + .5)
    ax.set_yticks(np.arange(0, len(names)) + .5)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticklabels(names, fontsize=18)
    ax.yaxis.set_ticklabels(names, fontsize=18)

    colors = [label_cmap[i] / 255.0 for i in range(len(names))]
    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
    plt.tight_layout()


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)


@hydra.main(config_path="configs", config_name="eval_config.yaml")
def my_app(cfg: DictConfig) -> None:

    pytorch_data_dir = cfg.pytorch_data_dir
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)

    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)

    for model_path in cfg.model_paths:
      
        torch.multiprocessing.set_sharing_strategy('file_system')
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model = LitUnsupervisedSegmenter.load_from_checkpoint(
            model_path,
            map_location=device,
            weights_only=False
        )
        print(OmegaConf.to_yaml(model.cfg))

        loader_crop = "center"

        test_dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name=model.cfg.dataset_name,
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, loader_crop),
            target_transform=get_transform(cfg.res, True, loader_crop),
            mask=True,
            cfg=model.cfg,
        )

        test_loader = DataLoader(
            test_dataset,
            cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=flexible_collate
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model.eval().to(device)

        par_model = torch.nn.DataParallel(model.net) if cfg.use_ddp else model.net

        saved_data = defaultdict(list)

        with Pool(cfg.num_workers + 5) as pool:
            for i, batch in enumerate(tqdm(test_loader)):

                with torch.no_grad():
                    img = batch["img"].to(device)
                    label = batch["label"].to(device)

                    # CHAOS-style forward (from Code 1)
                    feats, code1 = par_model(img)
                    feats, code2 = par_model(img.flip(dims=[3]))
                    code = (code1 + code2.flip(dims=[3])) / 2

                    # Test-time augmentation (flip averaging)
                    code = (code1 + code2.flip(dims=[3])) / 2

                    # Resize to match label
                    code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

                    # Predictions
                    linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)

                    # code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

                    # linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)

                    # CHAOS cluster format
                    cluster_loss, cluster_probs = model.cluster_probe(code, None)
                    cluster_probs = torch.log_softmax(cluster_probs, dim=1)

                    if cfg.run_crf and dense_crf is not None:
                        linear_preds = batched_crf(pool, img, linear_probs).argmax(1).to(device)
                        cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).to(device)
                    else:
                        linear_preds = linear_probs.argmax(1)
                        cluster_preds = cluster_probs.argmax(1)

                    model.test_linear_metrics.update(linear_preds, label)
                    model.test_cluster_metrics.update(cluster_preds, label)

                    saved_data["linear_preds"].append(linear_preds.cpu())
                    saved_data["cluster_preds"].append(cluster_preds.cpu())
                    saved_data["label"].append(label.cpu())
                    saved_data["img"].append(img.cpu())

        saved_data = {k: torch.cat(v, dim=0) for k, v in saved_data.items()}

        tb_metrics = {
            **model.test_linear_metrics.compute(),
            **model.test_cluster_metrics.compute(),
        }

        print("")
        print(model_path)
        print(tb_metrics)

        # Save ALL images (CHAOS-friendly)
        for i in range(len(saved_data["img"])):

            plot_img = (prep_for_plot(saved_data["img"][i]) * 255).numpy().astype(np.uint8)
            plot_label = (model.label_cmap[saved_data["label"][i]]).astype(np.uint8)

            Image.fromarray(plot_img).save(join(result_dir, "img", f"{i}.jpg"))
            Image.fromarray(plot_label).save(join(result_dir, "label", f"{i}.png"))

            plot_cluster = model.label_cmap[
                model.test_cluster_metrics.map_clusters(saved_data["cluster_preds"][i])
            ].astype(np.uint8)

            Image.fromarray(plot_cluster).save(join(result_dir, "cluster", f"{i}.png"))

        plot_cm(model.test_cluster_metrics.histogram, model.label_cmap, model.cfg)
        plt.show()


if __name__ == "__main__":
    prep_args()
    my_app()