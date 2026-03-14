import argparse
import shutil
from pathlib import Path


def copy_visualization_objects(video_root, object_root, destination_root):
    video_root = Path(video_root)
    object_root = Path(object_root)
    destination_root = Path(destination_root)

    for model_dir in sorted(video_root.iterdir()):
        if not model_dir.is_dir() or "model" not in model_dir.name:
            continue

        for video_path in sorted(model_dir.glob("*.mp4")):
            sequence_name = video_path.stem
            source_root = object_root / model_dir.name / sequence_name
            destination_seq_root = destination_root / model_dir.name / sequence_name
            destination_seq_root.mkdir(parents=True, exist_ok=True)

            if "omomo" in model_dir.name:
                shutil.copytree(
                    source_root / "objs_step_6_bs_idx_0",
                    destination_seq_root / "objs",
                    dirs_exist_ok=True,
                )
                continue

            shutil.copytree(
                source_root / "objs_step_10_bs_idx_0",
                destination_seq_root / "objs",
                dirs_exist_ok=True,
            )
            shutil.copytree(
                source_root / "ball_objs_step_10_bs_idx_0",
                destination_seq_root / "ball_objs",
                dirs_exist_ok=True,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Copy rendered mesh folders for selected DecHOI videos.")
    parser.add_argument("--video-root", required=True, help="Folder containing per-model mp4 results.")
    parser.add_argument("--object-root", required=True, help="Folder containing exported object mesh folders.")
    parser.add_argument("--destination-root", required=True, help="Destination folder for copied mesh assets.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    copy_visualization_objects(args.video_root, args.object_root, args.destination_root)
