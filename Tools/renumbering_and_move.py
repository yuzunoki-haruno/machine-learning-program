from pathlib import Path


def renumbering_and_move(input_path: Path, output_path: Path, base_name: str) -> None:
    count = len(list(output_path.glob("*.png")))
    for path in sorted(input_path.glob("*.png")):
        count += 1
        filename = output_path / (base_name + f"{count:06}.png")
        path.rename(filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_video_path", type=str)
    parser.add_argument("output_dir_path", type=str)
    parser.add_argument("base_name", type=str)
    args = parser.parse_args()

    input_path = Path(args.input_video_path)
    output_path = Path(args.output_dir_path)
    output_path.mkdir(exist_ok=True, parents=True)

    renumbering_and_move(input_path, output_path, args.base_name)

    # logging.info(f"{count} images saved to {output_path}.")
