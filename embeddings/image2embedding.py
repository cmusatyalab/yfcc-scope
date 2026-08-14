import argparse
import wids
import torch
import numpy as np
import contextlib
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile

# Prevent crashes when encountering truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

TOTAL_NUM = 10016544
YFCC_URL = "https://storage.cmusatyalab.org/yfcc100m/yfcc100m.json"
OUTPUT_DIR = "/home/ubuntu/yfcc-scope/embeddings"
DINOV3_REPO_DIR = "/home/ubuntu/dinov3"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and save image embeddings for the YFCC dataset")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["clip", "dinov3"],
        help="Embedding model (clip or dinov3)",
    )
    parser.add_argument("-s", "--start", type=int, default=0, help="Starting index for processing")
    return parser.parse_args()


def make_transform_dinov3(resize_size: int = 256):
    from torchvision.transforms import v2

    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def get_model(method: str):
    if method == "clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        dim = 512
        resize_size = 224
    elif method == "dinov3":
        model = torch.hub.load(
            DINOV3_REPO_DIR,
            "dinov3_vits16plus",
            source="local",
            weights=f"{DINOV3_REPO_DIR}/checkpoints/dinov3_vits16plus_pretrain_lvd1689m.pth",
        )
        preprocess = make_transform_dinov3()
        dim = 384
        resize_size = 256
    else:
        raise ValueError("Invalid method. Choose either 'clip' or 'dinov3'.")

    return model, preprocess, dim, resize_size


def compute_normalized_embedding(method, images):
    if method == "clip":
        embeddings = model.encode_image(images)
    elif method == "dinov3":
        embeddings = model(images)
    else:
        raise ValueError("Invalid method. Choose either 'clip' or 'dinov3'.")

    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


class YFCCPreprocessDataset(Dataset):
    def __init__(self, wids_ds, preprocess_fn, resize_size, start_idx, total_len):
        self.wids_ds = wids_ds
        self.preprocess_fn = preprocess_fn
        self.resize_size = resize_size
        self.start_idx = start_idx
        self.total_len = total_len

    def __len__(self):
        return self.total_len - self.start_idx

    def __getitem__(self, i):
        real_idx = self.start_idx + i
        try:
            img_tensor = self.preprocess_fn(self.wids_ds[real_idx][".jpg"])
        except Exception as e:
            # In case there are images totally corrupted, return a zero tensor with the same shape as the preprocess output
            img_tensor = torch.zeros((3, self.resize_size, self.resize_size), dtype=torch.float32)

        return img_tensor, real_idx


if __name__ == "__main__":
    args = parse_args()

    model, preprocess, dim, resize_size = get_model(args.method)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = model.to(device)
    model.eval()

    use_amp = device == "cuda"
    amp_ctx = torch.autocast(device) if use_amp else contextlib.nullcontext()

    output_path = Path(OUTPUT_DIR) / args.method / "embeddings.npy"

    mm = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=(TOTAL_NUM, dim))

    ds = wids.ShardListDataset(YFCC_URL)

    dataset = YFCCPreprocessDataset(ds, preprocess, resize_size, args.start, TOTAL_NUM)
    loader = DataLoader(dataset, batch_size=256, num_workers=8, pin_memory=(device == "cuda"), shuffle=False)

    for images, indices in tqdm(loader, total=len(loader)):
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), amp_ctx:
            embeddings = compute_normalized_embedding(args.method, images)

        mm[indices.numpy(), :] = embeddings.cpu().numpy().astype(np.float16, copy=False)
        mm.flush()

    mm.flush()
