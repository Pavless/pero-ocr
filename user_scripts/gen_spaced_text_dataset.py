import json
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List
from dataclasses import dataclass, asdict
import cv2

import numpy as np
from scipy.special import log_softmax
import pickle

from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.force_alignment import align_text
from pero_ocr.document_ocr.crop_engine import EngineLineCropper
from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR

@dataclass
class DatasetEntry:
    id: str
    transcription: str
    char_positions: List[int]

class LabelEncoder:
    def __init__(self, char_to_idx: Dict[str, int]):
        self.char_to_idx = char_to_idx

    def encode(self, label: str):
        return np.array([self.char_to_idx[c] for c in label])

class DatasetBuilder:

    def __init__(self,
                 xml_path: Path,
                 image_path: Path,
                 dataset_path: Path,
                 ocr_engine: PytorchEngineLineOCR,
                 line_cropper: EngineLineCropper,
                 label_encoder: LabelEncoder,
                 verbose=False):
        self.xml_path = xml_path
        self.image_path = image_path
        self.dataset_path = dataset_path

        self.ocr_engine = ocr_engine
        self.line_cropper = line_cropper
        self.label_encoder = label_encoder

        self.verbose = verbose

        self.dataset: List[DatasetEntry] = []

    def process_page(self, page_id: str):
        image = cv2.imread(str(self.image_path / f"{page_id}.jpg"))
        page_layout = PageLayout(file=str(self.xml_path / f"{page_id}.xml"))

        for line in page_layout.lines_iterator():
            if line.transcription is None:
                if self.verbose:
                    print(f"skipping: {page_id}-{line.id}")
                continue
            line_crop = self.line_cropper.crop(image, line.baseline, line.heights)
            line_batched = line_crop[np.newaxis]

            _, logits, _ = self.ocr_engine.process_lines(line_batched, sparse_logits=False, tight_crop_logits=True)
            logits = logits[0]
            log_probs = log_softmax(logits, axis=1)
            labels = self.label_encoder.encode(line.transcription)
            char_positions = align_text(-log_probs, labels, log_probs.shape[1] - 1)
            char_positions = char_positions * self.ocr_engine.net_subsampling

            dataset_entry = DatasetEntry(f"{page_id}-{line.id}",
                                         line.transcription,
                                         [i for i in char_positions])
            self.dataset.append(dataset_entry)

            if self.verbose:
                for i, p in enumerate(char_positions):
                    color = [0,0,255] if line.transcription[i] != " " else [0,255,0]
                    line_crop[:, p, :] = color
                cv2.imshow("Character positions", line_crop)
                cv2.waitKey(0)


    def save(self):
        dataset = [asdict(item) for item in self.dataset]
        print(f"Saving, dataset size: {len(dataset)}...")

        if self.dataset_path is not None:
            with open(self.dataset_path, "wb") as f:
                pickle.dump(dataset, f)

def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--ocr", help="The OCR .json config file", type=Path, required=True)
    parser.add_argument("-x", "--xml-path", help="The path to the directory containing .xml page files", type=Path, required=True)
    parser.add_argument("-i", "--image-path", help="The path to the directory containing page images", type=Path, required=True)
    parser.add_argument("-d", "--dataset-path", help="The path to the dataset output file", type=Path, default=None)
    parser.add_argument("-v", "--verbose", help="If specified, the individual character positions for each line are displayed", action="store_true")
    args = parser.parse_args()


    with open(args.ocr, "r") as f:
        ocr_config = json.load(f)

    char_to_idx = {c: i for i, c in enumerate(ocr_config["characters"])}

    ocr_engine = PytorchEngineLineOCR(args.ocr, gpu_id=0)
    line_cropper = EngineLineCropper(line_height=ocr_config["line_px_height"], poly=2, scale=ocr_config["line_vertical_scale"])
    label_encoder = LabelEncoder(char_to_idx)
    dataset_builder = DatasetBuilder(args.xml_path,
                                     args.image_path,
                                     args.dataset_path,
                                     ocr_engine,
                                     line_cropper,
                                     label_encoder,
                                     verbose=args.verbose)
    
    page_ids = [f.stem for f in args.xml_path.glob("*.xml")]
    total_count = len(page_ids)
    for i, page_id in enumerate(page_ids):
        dataset_builder.process_page(page_id)
        print("                                                         \r", end="")
        print(f"Progress: {i+1}/{total_count} ({(i+1)/total_count*100:.2f} %)", end="")
    print()
    dataset_builder.save()

if __name__ == "__main__":
    main()
