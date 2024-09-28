import logging
from pathlib import Path

import cv2
import tqdm


def video_to_images(input_path: Path, output_path: Path, fps: float) -> int:
    count = 0

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logging.error(f"Cannot open {input_path}")
        return count

    n_mod = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm.tqdm(range(n_frame)):
        success, frame = cap.read()

        if not success:
            logging.warning("Cannot capture. Exiting ...")
            return count

        if i % n_mod == 0:
            count += 1
            filename = output_path / f"image{count:06}.png"
            cv2.imwrite(str(filename), frame)

    cap.release()

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_video_path", type=str)
    parser.add_argument("output_dir_path", type=str)
    parser.add_argument("fps", type=float)
    args = parser.parse_args()

    input_path = Path(args.input_video_path)
    output_path = Path(args.output_dir_path)
    output_path.mkdir(exist_ok=True, parents=True)

    count = video_to_images(input_path, output_path, args.fps)

    logging.info(f"{count} images saved to {output_path}.")
