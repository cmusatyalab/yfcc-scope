import wids
import torch
import open_clip
import numpy as np
import contextlib
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile

# Prevent crashes when encountering truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

TOTAL_NUM = 10016544
YFCC_URL = "https://storage.cmusatyalab.org/yfcc100m/yfcc100m.json"
OUTPUT_PATH = "/home/ubuntu/yfcc-scope/clip-embedding/yfcc_image_embeddings.npy"

ds = wids.ShardListDataset(YFCC_URL)

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer("ViT-B-32")


# Batched processing using DataLoader
def compute_and_save_image_embeddings(
    ds,
    model,
    preprocess,
    total_num,
    batch_size=256,
    start=0,
    dtype=np.float16,
    save_every_batches=1,
    existing_memmap=None,
    num_workers=8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = model.to(device)
    model.eval()

    use_amp = device == "cuda"
    amp_ctx = torch.autocast(device) if use_amp else contextlib.nullcontext()

    # Determine embedding dimension with a single sample
    sample = preprocess(ds[start][".jpg"]).unsqueeze(0).to(device)
    with torch.no_grad(), amp_ctx:
        dim = model.encode_image(sample).shape[-1]

    if existing_memmap is None:
        mm = np.lib.format.open_memmap(OUTPUT_PATH, mode="w+", dtype=dtype, shape=(total_num, dim))
    else:
        mm = existing_memmap

    class YFCCPreprocessDataset(Dataset):
        def __init__(self, wids_ds, preprocess_fn, start_idx, total_len):
            self.wids_ds = wids_ds
            self.preprocess_fn = preprocess_fn
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
                img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

            return img_tensor, real_idx

    dataset = YFCCPreprocessDataset(ds, preprocess, start, total_num)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(device == "cuda"), shuffle=False
    )

    batch_count = 0
    for images, indices in tqdm(loader, total=len(loader)):
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), amp_ctx:
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        mm[indices.numpy(), :] = feats.cpu().numpy().astype(dtype, copy=False)

        batch_count += 1
        if batch_count % save_every_batches == 0:
            mm.flush()

    mm.flush()
    return mm


mm = compute_and_save_image_embeddings(
    ds=ds,
    model=model,
    preprocess=preprocess,
    total_num=TOTAL_NUM,
    batch_size=256,
    start=0,
    dtype=np.float16,
    num_workers=8,
)
