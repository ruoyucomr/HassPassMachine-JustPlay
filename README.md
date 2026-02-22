# HashPass Machine (CPU + CUDA)

## Prerequisites
- Rust toolchain (stable)
- Visual Studio 2022 Build Tools (C++ workload)
- CMake
- NVIDIA CUDA Toolkit (tested with CUDA 13.1)

## Build
### CPU
```powershell
C:\Users\ruoyu\.cargo\bin\cargo.exe build --release
```

### CUDA (GPU)
```powershell
# Optional: set CUDA_PATH if not already present
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
$env:Path="C:\Program Files\CMake\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;" + $env:Path

C:\Users\ruoyu\.cargo\bin\cargo.exe build --release --features cuda
```

## Run
### CPU
```powershell
.\target\release\hashpass-machine.exe
```

### CUDA (GPU)
```powershell
.\target\release\hashpass-machine.exe --gpu --gpu-device=0 --gpu-batch=16 --max-hps=29.9
```

Notes:
- `--gpu-batch` controls GPU memory usage. Rough estimate: `mem_MB * batch`.
  - Example: `mem=256MB, batch=16` => ~4GB.
- `--max-hps` caps throughput to avoid server limits (e.g. 29.9 H/s).
- If you see `os error 10048`, the port is in use. Kill the old process or:
```powershell
.\target\release\hashpass-machine.exe --port=19527
```
Then update `bridge.js`:
```js
const RUST_URL = "http://localhost:19527/";
```

## Browser Bridge
1. Keep the Rust process running.
2. Open the target page in your browser.
3. Open DevTools Console and paste `bridge.js` from the repo root.
   - You may need to type `allow pasting` in the console first.
4. Allow popups for the site so the relay window can open.

Once connected, the Rust console should show:
- `WebSocket 连接`
- `已发送 ready`
- `开始挖矿`
