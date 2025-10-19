from pathlib import Path
import re
import shutil


def move_and_rename_files(src_folder, dst_folder, start_index):
    """
    Move files from src_folder with an index >= start_index into dst_folder
    and renumber them so the sequence continues after the current maximum in
    the destination folder.
    """
    src = Path(src_folder)
    dst = Path(dst_folder)
    dst.mkdir(parents=True, exist_ok=True)

    # Capture the numeric index from the filename, e.g., image_001.jpg -> 1
    pattern = re.compile(r"(\d+)")

    def get_index(path):
        match = pattern.search(path.stem)
        return int(match.group(1)) if match else None

    # Collect all numbered files in the source folder
    src_files = [f for f in src.iterdir() if f.is_file() and get_index(f) is not None]
    src_files.sort(key=get_index)

    # Determine the highest index already present in the destination folder
    dst_files = [f for f in dst.iterdir() if f.is_file() and get_index(f) is not None]
    max_dst_index = max([get_index(f) for f in dst_files], default=0)

    # Select source files whose index is >= start_index
    files_to_move = [f for f in src_files if get_index(f) >= start_index]

    next_index = max_dst_index + 1

    for f in files_to_move:
        new_name = pattern.sub(f"{next_index}", f.stem) + f.suffix
        new_path = dst / new_name

        shutil.move(str(f), str(new_path))
        print(f"Moved {f.name} -> {new_path.name}")
        next_index += 1
    print(f"Moved {len(files_to_move)} file(s).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Move and rename files based on index."
    )
    parser.add_argument("src_folder", type=str, help="Source folder path")
    parser.add_argument("dst_folder", type=str, help="Destination folder path")
    parser.add_argument("start_index", type=int, help="Start index for moving files")

    args = parser.parse_args()
    move_and_rename_files(
        src_folder=args.src_folder,
        dst_folder=args.dst_folder,
        start_index=args.start_index,
    )
# Example usage:
# move_and_rename_files("path/to/src", "path/to/dst", 10)
