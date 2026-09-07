# Apple Silicon Docker training environment

This environment runs natively as `linux/arm64` on Apple Silicon Macs,
including M5. It uses PyTorch on CPU because Docker Desktop and Colima do not
expose the macOS Metal/MPS GPU to Linux containers.

## Start the Docker engine

This machine uses Docker CLI with Colima. Start it after a reboot with:

```bash
colima start
```

## Build and verify

From the repository root:

```bash
docker compose -f compose.apple-silicon.yaml build sar-apple
docker compose -f compose.apple-silicon.yaml run --rm sar-apple \
  python docker/verify_apple_silicon.py
```

The verification performs an actual forward pass, backward pass, and Adam
optimizer step with `TransSARV2_DualFreqNG_Bottle`.

## Enter the environment

```bash
docker compose -f compose.apple-silicon.yaml run --rm sar-apple
```

The repository is mounted at `/workspace`, so datasets and experiment outputs
remain on the Mac. For training in this container, explicitly select CPU:

```bash
python train_reproducible.py --device cpu [other arguments]
```

Stop the Docker VM when it is no longer needed:

```bash
colima stop
```
