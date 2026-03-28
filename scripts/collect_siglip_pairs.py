#!/usr/bin/env python3
"""
Collect paired (SigLIP embedding, LeWM embedding) data for projection training.

Takes goal images from the dataset, encodes them through both the LeWM ViT
encoder and a SigLIP encoder, and saves the paired embeddings. The output
is consumed by train_vlm_projection.py.

Usage (on-pod with GPU):
    python scripts/collect_siglip_pairs.py --policy tworoom/lewm --config-name tworoom
    python scripts/collect_siglip_pairs.py --policy pusht/lejepa --config-name pusht --num-samples 5000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm

from harness.pipeline import PlanningPipeline


def main():
    parser = argparse.ArgumentParser(description="Collect SigLIP→LeWM paired embeddings")
    parser.add_argument("--policy", default="tworoom/lewm")
    parser.add_argument("--config-name", default="tworoom")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="Number of goal images to encode")
    parser.add_argument("--siglip-model", default="ViT-B-16-SigLIP",
                        help="SigLIP model name for open_clip")
    parser.add_argument("--siglip-pretrained", default="webli",
                        help="SigLIP pretrained weights")
    parser.add_argument("--output", default=None,
                        help="Output path (default: siglip_pairs_{config}.pt)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"siglip_pairs_{args.config_name}.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load eval config
    from hydra import compose, initialize_config_dir
    config_dir = str(Path("./config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name)

    # Load dataset
    cache_dir = Path(swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )

    # Load LeWM pipeline (for ViT encoding)
    print("Loading LeWM pipeline...")
    pipeline = PlanningPipeline(args.policy)
    pipeline.warmup()

    # Load SigLIP model
    print("Loading SigLIP model...")
    import open_clip
    siglip_model, _, siglip_preprocess = open_clip.create_model_and_transforms(
        args.siglip_model, pretrained=args.siglip_pretrained
    )
    siglip_model = siglip_model.to(device).eval()
    siglip_model.requires_grad_(False)

    # Sample random goal images from dataset
    rng = np.random.default_rng(args.seed)
    n_rows = len(dataset)
    sample_indices = rng.choice(n_rows, size=min(args.num_samples, n_rows), replace=False)
    sample_indices = np.sort(sample_indices)

    print(f"Encoding {len(sample_indices)} goal images...")

    lewm_embeddings = []
    siglip_embeddings = []

    for i, idx in enumerate(sample_indices):
        row = dataset.get_row_data(int(idx))
        image = row["pixels"]
        if isinstance(image, np.ndarray) and image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        # LeWM ViT encoding → (1, 1, 192) → (192,)
        with torch.inference_mode():
            tensor = pipeline.preprocess(image)
            lewm_emb = pipeline.encode(tensor).squeeze().cpu()  # (192,)

        # SigLIP encoding → (768,) or similar
        # SigLIP expects PIL-like input through its preprocess transform
        from PIL import Image
        pil_image = Image.fromarray(image)
        siglip_input = siglip_preprocess(pil_image).unsqueeze(0).to(device)
        with torch.inference_mode():
            siglip_emb = siglip_model.encode_image(siglip_input).squeeze().cpu().float()

        lewm_embeddings.append(lewm_emb)
        siglip_embeddings.append(siglip_emb)

        if (i + 1) % 200 == 0 or i == 0:
            print(f"  {i+1}/{len(sample_indices)} encoded")

    # Stack into tensors
    lewm_tensor = torch.stack(lewm_embeddings)      # (N, 192)
    siglip_tensor = torch.stack(siglip_embeddings)   # (N, siglip_dim)

    print(f"\nLeWM embeddings: {lewm_tensor.shape}")
    print(f"SigLIP embeddings: {siglip_tensor.shape}")

    # Save in the format expected by train_vlm_projection.py
    save_dict = {
        "vlm_features": siglip_tensor,         # (N, siglip_dim)
        "target_embeddings": lewm_tensor,       # (N, 192)
        "source": "siglip",
        "siglip_model": args.siglip_model,
        "siglip_pretrained": args.siglip_pretrained,
        "policy": args.policy,
        "config": args.config_name,
        "num_samples": len(sample_indices),
    }

    torch.save(save_dict, args.output)
    print(f"Saved to {args.output}")
    print(f"\nNext step: train the projection with:")
    print(f"  python scripts/train_vlm_projection.py --source siglip --data {args.output}")


if __name__ == "__main__":
    main()
