import torch
import torch.multiprocessing as mp
import signal
import sys
import os
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GPU%(gpu_id)s] %(message)s",
    datefmt="%H:%M:%S"
)

def stress_gpu(gpu_id):
    # Detach from parent process group so Ray's cleanup doesn't kill us
    os.setsid()

    # Handle signals gracefully in child
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    log = logging.LoggerAdapter(logging.getLogger(), {"gpu_id": gpu_id})

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    log.info(f"Starting GPU stress on GPU {gpu_id}")

    size = 512
    iteration = 0

    while True:
        try:
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            iteration += 1

            if iteration % 1000 == 0:
                log.info(f"GPU {gpu_id} - iteration {iteration}")

        except torch.cuda.OutOfMemoryError:
            log.warning(f"GPU {gpu_id} OOM — reducing matrix size and retrying")
            size = max(64, size // 2)
            torch.cuda.empty_cache()
            time.sleep(1)

        except Exception as e:
            log.error(f"GPU {gpu_id} error: {e}, restarting loop in 2s")
            time.sleep(2)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs")

    processes = []
    for gpu_id in range(gpu_count):
        p = mp.Process(target=stress_gpu, args=(gpu_id,), daemon=False)
        p.start()
        processes.append(p)
        print(f"Spawned PID {p.pid} for GPU {gpu_id}")

    # Save PIDs so you can kill them manually later
    pid_file = "/tmp/burn_gpu.pids"
    with open(pid_file, "w") as f:
        f.write("\n".join(str(p.pid) for p in processes))
    print(f"PIDs saved to {pid_file} — kill with: kill $(cat {pid_file})")

    # Parent handles signals but doesn't propagate to children (they're in own session)
    def shutdown(sig, frame):
        print("\nShutting down burn_gpu workers...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()