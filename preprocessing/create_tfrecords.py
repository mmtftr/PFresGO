#!/usr/bin/env python3
import argparse
from pathlib import Path
from dataset_processor import DatasetConfig, SequenceProcessor

def main():
    parser = argparse.ArgumentParser(description="Convert protein sequences to TFRecord format")
    parser.add_argument("--annotation-file", type=Path, required=True,
                      help="Path to annotation TSV file")
    parser.add_argument("--protein-list", type=Path, required=True,
                      help="Path to protein list file")
    parser.add_argument("--sequence-file", type=Path, required=True,
                      help="Path to FASTA sequence file")
    parser.add_argument("--output-prefix", type=Path, required=True,
                      help="Prefix for output TFRecord files")
    parser.add_argument("--num-threads", type=int, default=3,
                      help="Number of threads to use")
    parser.add_argument("--num-shards", type=int, default=3,
                      help="Number of shards to create")

    args = parser.parse_args()

    config = DatasetConfig(
        annotation_file=args.annotation_file,
        protein_list_file=args.protein_list,
        sequence_file=args.sequence_file,
        tfrecord_prefix=args.output_prefix,
        num_threads=args.num_threads,
        num_shards=args.num_shards
    )

    processor = SequenceProcessor(config)
    processor.process_to_tfrecord()

if __name__ == "__main__":
    main()