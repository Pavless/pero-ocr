import json
import os
import time
import argparse
from typing import List
from attr import dataclass

import torch
import torch.nn.functional as F
import numpy as np
import pickle

from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.force_alignment import force_align, align_text

## Augmented original function greedy_decode_ctc
## from pero_ocr/ocr_engine/pytorch_ocr_engine.py
# scores_probs should be N,C,T, blank is last class

def greedy_decode_ctc(logits, characters, keep_blank=True):
    if len(logits.shape) == 2:
        logits = np.concatenate((logits[:, 0:1], logits), axis=1)
        logits[:, 0] = -1000
        logits[-1, 1] = 1000
    else:
        logits = np.concatenate((logits[:, :, 0:1], logits), axis=2)
        logits[:, :, 0] = -1000
        logits[:, -1, 0] = 1000

    best = np.argmax(logits, 1) + 1
    mask = best[:, :-1] == best[:, 1:]
    best = best[:, 1:]
    best[mask] = 0
    best[best == logits.shape[1]] = 0
    best = best - 1
    outputs = []
    for line in best:
        if keep_blank:
            outputs.append(''.join([characters[c] if c >= 0 else "~" for c in line]))
        else:
            outputs.append(''.join([characters[c] if c >= 0 else "" for c in line]))

    return outputs

def encode_transcription(text, char_to_idx):
    return [char_to_idx[c]+1 for c in text]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--input-xml-path", required=True)
    parser.add_argument("-o", "--ocr", required=True)
    parser.add_argument("-l", "--input-logit-path", required=True)
    parser.add_argument("-d", "--output-dataset-path", default=None)
    args = parser.parse_args()
    return args

@dataclass
class DatasetEntry:
    id: str
    best_ctc_path: str

class DatasetBuilder:

    def __init__(self, input_xml_path, input_logit_path, output_dataset_path, char_to_idx, idx_to_char):
        self.input_xml_path = input_xml_path
        self.input_logit_path = input_logit_path
        self.output_dataset_path = output_dataset_path
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

        self.paths: List[DatasetEntry] = []
        self.characters = None

    def process_page(self, file_id, index, ids_count):
        print(f"Processing {file_id}")
        t1 = time.time()
        page_layout = PageLayout(file=os.path.join(self.input_xml_path, file_id + '.xml'))
        page_layout.load_logits(os.path.join(self.input_logit_path, file_id + '.logits'))

        for idx_r, region in enumerate(page_layout.regions):
            for idx_l, line in enumerate(region.lines):
                print("@@@@@")
                print(line.transcription)
                self.characters = line.characters
                logits = torch.tensor(line.logits.toarray())
                logits = logits.unsqueeze(0)
                logits = logits.permute(0, 2, 1)
                log_probs = F.log_softmax(logits, dim=1)

                log_probs = torch.squeeze(log_probs).numpy().T
                best_ctc_path = force_align(-log_probs, encode_transcription(line.transcription, self.char_to_idx), 511)

                best_ctc_path = np.array(best_ctc_path) - 1
                # best_ctc_path = best_ctc_path[np.nonzero(best_ctc_path >= 0)]
                print(''.join([self.idx_to_char[c] if c >= 0 else "~" for c in best_ctc_path]))
                # exit()

                # print(align_text(-log_probs, np.asarray(encode_transcription(line.transcription, self.char_to_idx)) , 0))

                best_ctc_path = greedy_decode_ctc(np.expand_dims(line.logits.toarray().T, 0), line.characters)
                best_ctc_path = np.squeeze(best_ctc_path)
                print(best_ctc_path)

                best_ctc_path = greedy_decode_ctc(np.expand_dims(line.logits.toarray().T, 0), line.characters, keep_blank=False)
                best_ctc_path = np.squeeze(best_ctc_path)
                print(best_ctc_path)
                best_ctc_path = str(best_ctc_path)

                best_ctc_path = force_align(-log_probs, encode_transcription(best_ctc_path, self.char_to_idx), 0)

                best_ctc_path = np.array(best_ctc_path) - 1
                # best_ctc_path = best_ctc_path[np.nonzero(best_ctc_path >= 0)]
                print(''.join([self.idx_to_char[c] if c >= 0 else "~" for c in best_ctc_path]))
                
                self.paths.append(DatasetEntry(f"{file_id}-{line.id}", best_ctc_path))
            exit()
        print("DONE {current}/{total} ({percentage:.2f} %) [id: {file_id}] Time:{time:.2f}".format(
            current=index + 1, total=ids_count, percentage=(index + 1) / ids_count * 100,
            file_id=file_id, time=time.time() - t1))

    def save(self):
        paths = {}
        for entry in self.paths:
            paths[entry.id] = entry.best_ctc_path

        dataset = {
            "characters": self.characters,
            "paths": paths
        }

        if self.output_dataset_path is not None:
            with open(self.output_dataset_path, "wb") as f:
                pickle.dump(dataset, f)

def main():
    args = parse_arguments()

    input_xml_path = args.input_xml_path
    input_logit_path = args.input_logit_path
    output_dataset_path = args.output_dataset_path

    with open(args.ocr, "r") as f:
        ocr_config = json.load(f)

    char_to_idx = {c: i for i, c in enumerate(ocr_config["characters"])}

    dataset_builder = DatasetBuilder(input_xml_path, input_logit_path, output_dataset_path, char_to_idx, ocr_config["characters"])

    logits_to_process = [f for f in os.listdir(input_logit_path)
                         if os.path.splitext(f)[1] ==  ".logits"]
    ids_to_process = [os.path.splitext(os.path.basename(file))[0] for file in logits_to_process]

    for index, file_id in enumerate(ids_to_process):
        dataset_builder.process_page(file_id, index, len(ids_to_process))
    
    dataset_builder.save()

if __name__ == "__main__":
    main()
