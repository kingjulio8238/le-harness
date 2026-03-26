# D1: Upload Multi-Task Data to Pod

Google Drive rate-limits automated downloads (gdown). Download manually in your browser and upload via SCP.

## 1. Download in your browser

Go to each folder and download `lejepa.tar.zst` from the `ckpt/` subfolder:

- **TwoRoom:** https://drive.google.com/drive/folders/1-OmNHgZLU19EBUNAqPK4Y_SeJehVO11f → `ckpt/` → `lejepa.tar.zst`
- **Reacher:** https://drive.google.com/drive/folders/1UwTAyn1yZg7oiiiOjMceLcOCpkQlUqKp → `ckpt/` → `lejepa.tar.zst`
- **Cube:** https://drive.google.com/drive/folders/1IGrSf5cOGY79jLkJlVCS10MvHV6F62bB → `ckpt/` → `lejepa.tar.zst`

Also download datasets for Reacher and Cube:
- **Reacher:** `dataset/` → `reacher.tar.zst`
- **Cube:** `dataset/` → `cube_single_expert.tar.zst`

(We already have `tworoom.h5` on the volume)

## 2. Upload to pod via SCP

From your local terminal (not the pod), rename each file to avoid conflicts, then:

```bash
# Get your pod's SSH info from RunPod dashboard (SSH over exposed TCP)
# Format: ssh root@<POD_IP> -p <PORT> -i <KEY>

# Upload checkpoints
scp -P <PORT> tworoom_lejepa.tar.zst root@<POD_IP>:/tmp/
scp -P <PORT> reacher_lejepa.tar.zst root@<POD_IP>:/tmp/
scp -P <PORT> cube_lejepa.tar.zst root@<POD_IP>:/tmp/

# Upload datasets
scp -P <PORT> reacher.tar.zst root@<POD_IP>:/tmp/
scp -P <PORT> cube_single_expert.tar.zst root@<POD_IP>:/tmp/
```

## 3. Decompress on pod

Once uploaded, run on the pod:

```bash
# TwoRoom checkpoint
mkdir -p /workspace/data/tworoom
tar --zstd -xvf /tmp/tworoom_lejepa.tar.zst -C /workspace/data/tworoom/

# Reacher checkpoint + dataset
mkdir -p /workspace/data/dmc/reacher
tar --zstd -xvf /tmp/reacher_lejepa.tar.zst -C /workspace/data/dmc/reacher/
tar --zstd -xvf /tmp/reacher.tar.zst -C /workspace/data/

# Cube checkpoint + dataset
mkdir -p /workspace/data/ogb_cube
tar --zstd -xvf /tmp/cube_lejepa.tar.zst -C /workspace/data/ogb_cube/
tar --zstd -xvf /tmp/cube_single_expert.tar.zst -C /workspace/data/

# Clean up
rm /tmp/*.tar.zst
```

## 4. Verify

```bash
ls /workspace/data/tworoom/lejepa_object.ckpt
ls /workspace/data/dmc/reacher/lejepa_object.ckpt
ls /workspace/data/ogb_cube/lejepa_object.ckpt
ls /workspace/data/tworoom.h5
ls /workspace/data/reacher*.h5
ls /workspace/data/cube_single_expert*.h5
```

**Note:** The checkpoint files inside the tar might be named differently (e.g., just `checkpoint.ckpt`). After extracting, check what's inside and rename to `lejepa_object.ckpt` if needed.

## 5. Run D1 benchmarks

Once data is in place:

```bash
export STABLEWM_HOME=/workspace/data
export MUJOCO_GL=egl
cd /workspace/le-harness

# Flat CEM on each task
python eval.py --config-name=tworoom policy=tworoom/lejepa
python eval.py --config-name=cube policy=ogb_cube/lejepa
python eval.py --config-name=reacher policy=dmc/reacher/lejepa

# Dream Tree vs flat CEM on each task
python scripts/eval_dream_tree.py --policy tworoom/lejepa --mode both
python scripts/eval_dream_tree.py --policy ogb_cube/lejepa --mode both
python scripts/eval_dream_tree.py --policy dmc/reacher/lejepa --mode both
```
