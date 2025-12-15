import os
import glob
import random
from typing import List, Tuple, Dict

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def _glob_images(root: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root, ext)))
    return sorted(paths)


def collect_mvtec_binary(repo_root: str, categories: List[str]) -> Tuple[List[str], List[int], Dict[int, str]]:

    data_root = os.path.join(repo_root, "data", "mvtec")

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"MVTec root not found at: {data_root}")

    all_paths, all_labels = [], []

    for cat in categories:
        cat_root = os.path.join(data_root, cat)

        # good images from train and test
        if not os.path.isdir(cat_root):
            raise FileNotFoundError(
                f"Category root not found for '{cat}' at path {cat_root!r}."
            )
        good_train = _glob_images(os.path.join(cat_root, "train", "good"))
        good_test  = _glob_images(os.path.join(cat_root, "test", "good"))

        # defect images are all subfolders in test except 'good'
        defect_paths = []
        test_root = os.path.join(cat_root, "test")
        if not os.path.isdir(test_root):
            raise FileNotFoundError(
                f"Expected test folder for category '{cat}' not found at {test_root!r}."
            )
        for sub in os.listdir(test_root):
            if sub == "good":
                continue
            defect_paths.extend(_glob_images(os.path.join(test_root, sub)))

        all_paths.extend(good_train + good_test)
        all_labels.extend([0] * (len(good_train) + len(good_test)))

        all_paths.extend(defect_paths)
        all_labels.extend([1] * len(defect_paths))

    label_map = {0: "no_defect", 1: "defect"}
    return all_paths, all_labels, label_map


def stratified_split(paths: List[str], labels: List[int], seed: int = 0, train=0.7, val=0.15):
    """
    Simple stratified split by label.
    """
    assert len(paths) == len(labels)
    rnd = random.Random(seed)

    idx_by_label = {0: [], 1: []}
    for i, y in enumerate(labels):
        idx_by_label[y].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for y, idxs in idx_by_label.items():
        rnd.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train)
        n_val = int(n * val)
        train_idx += idxs[:n_train]
        val_idx += idxs[n_train:n_train + n_val]
        test_idx += idxs[n_train + n_val:]

    rnd.shuffle(train_idx); rnd.shuffle(val_idx); rnd.shuffle(test_idx)

    def pack(idxs):
        return [paths[i] for i in idxs], [labels[i] for i in idxs]

    return pack(train_idx), pack(val_idx), pack(test_idx)
