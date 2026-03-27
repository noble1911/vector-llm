"""Quick benchmark: how fast can we pull frames from Vector's camera?

Tests streaming mode via camera feed.
Usage: python scripts/camera_bench.py
"""

import time
from io import BytesIO

import anki_vector
from anki_vector.exceptions import VectorPropertyValueNotReadyException


def main():
    print("Connecting to Vector...")
    robot = anki_vector.Robot(
        cache_animation_lists=False,
        behavior_activation_timeout=30,
    )
    robot.connect()
    print("Connected.\n")

    try:
        # Start streaming feed
        robot.camera.init_camera_feed()
        print("Waiting for camera feed to start...")
        for _ in range(100):
            try:
                img = robot.camera.latest_image
                if img is not None:
                    break
            except VectorPropertyValueNotReadyException:
                pass
            time.sleep(0.05)
        else:
            print("ERROR: No frames received after 5s")
            return

        print(f"First frame received: {img.raw_image.size}\n")

        # === Stream latency benchmark ===
        print("=== Stream mode (20 frames) ===")
        last_id = img.image_id
        times = []
        for i in range(20):
            t0 = time.perf_counter()
            while True:
                try:
                    img = robot.camera.latest_image
                    if img is not None and img.image_id != last_id:
                        break
                except VectorPropertyValueNotReadyException:
                    pass
                time.sleep(0.002)
            elapsed = time.perf_counter() - t0
            last_id = img.image_id
            times.append(elapsed)
            print(f"  frame {i+1}: {elapsed*1000:.0f}ms  ({img.raw_image.size})")

        avg = sum(times) / len(times)
        print(f"\n  avg: {avg*1000:.0f}ms  min: {min(times)*1000:.0f}ms  max: {max(times)*1000:.0f}ms")
        print(f"  ~{1/avg:.1f} fps effective\n")

        # === JPEG encode benchmark ===
        print("=== JPEG encode ===")
        pil_img = img.raw_image
        for quality in [50, 75, 95]:
            t0 = time.perf_counter()
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            elapsed = time.perf_counter() - t0
            size_kb = buf.tell() / 1024
            print(f"  JPEG q={quality}: {elapsed*1000:.1f}ms  {size_kb:.0f}KB")

        # === Single capture benchmark ===
        print("\n=== Single capture (5 shots) ===")
        times = []
        for i in range(5):
            t0 = time.perf_counter()
            image = robot.camera.capture_single_image()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  capture {i+1}: {elapsed*1000:.0f}ms  ({image.raw_image.size})")

        avg = sum(times) / len(times)
        print(f"\n  avg: {avg*1000:.0f}ms  min: {min(times)*1000:.0f}ms  max: {max(times)*1000:.0f}ms")

    finally:
        robot.disconnect()

    print("\nDone.")


if __name__ == "__main__":
    main()
