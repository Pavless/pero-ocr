import itertools
from math import ceil
from pathlib import  Path
from argparse import ArgumentParser
import pickle
import time
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont


def load_ctc_paths(path: Path) -> Tuple[List[str], Dict[str, List[int]]]:
    with open(path, "rb") as f:
        ctc_paths = pickle.load(f)
    return ctc_paths["characters"], ctc_paths["paths"]

def process_line(line: Image.Image, ctc_path: List[int], characters: List[str]):
    for i, c in enumerate(characters):
        if c == " ":
            characters[i]="␣"

    padding = 0
    width, height = line.size

    fix_width = int(ceil(width / 32) * 32) - width
    width = int(ceil(width / 32) * 32) + 2*padding



    last_i = 0
    for i, c in enumerate(ctc_path):
        if c != -1:
            last_i = i

    ctc_path = ctc_path[:last_i+1]
    ctc_path = list(ctc_path)

    to_add = len(ctc_path) * ((width + fix_width)/width - 1)+5
    ctc_path.extend([-1 for _ in range(round(to_add))])

    dx = width / len(ctc_path)


    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)

    new_img = Image.new(line.mode, (width, height * 2), color=(0,0,0))
    new_img.paste(line, (padding,0))

    draw = ImageDraw.Draw(new_img)
    draw.line((0,height,width, height), fill=(0,255,0))
    for i, c in enumerate(ctc_path):
        if c != -1:
            x = round((i + .5) * dx)
            # -5 to center character
            # because x-5 is the left edge of the character
            draw.text((x-5, height + 12), characters[c], font=fnt, fill=(255,255,255))
            if characters[c] == "␣":
                draw.line((x, 0, x, height), fill=(0,0,128))
    
    return new_img


def main():
    parser = ArgumentParser()
    parser.add_argument("-l", "--lines", type=Path, required=True)
    parser.add_argument("-p", "--ctc-paths", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    characters, ctc_paths = load_ctc_paths(args.ctc_paths)

    total_count = len(ctc_paths.items())
    for i, (id, ctc_path) in enumerate(itertools.islice(ctc_paths.items(), 100)):
        line_img = Image.open(args.lines / f"{id}.jpg")
        visualization_img = process_line(line_img, ctc_path, characters)
        visualization_img.save(output / f"{id}.jpg")
        print("                                                         \r", end="")
        print(f"Progress: {i+1}/{total_count} ({(i+1)/total_count*100:.2f} %)", end="")
    print()

if __name__ == "__main__":
    main()