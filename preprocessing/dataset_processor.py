from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
import numpy as np
import h5py
from Bio import SeqIO
import gzip
import csv
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    annotation_file: Path
    protein_list_file: Path
    sequence_file: Path
    tfrecord_prefix: Path
    embeddings_file: Path
    num_threads: int = 3
    num_shards: int = 3
    min_seq_length: int = 60
    max_seq_length: int = 1000

    def __post_init__(self):
        """Validate configuration parameters"""
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        if not self.protein_list_file.exists():
            raise FileNotFoundError(f"Protein list file not found: {self.protein_list_file}")
        if not self.sequence_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {self.sequence_file}")
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive")

class SequenceProcessor:
    """Process protein sequences and convert to TFRecord format"""

    VALID_AMINO_ACIDS = {'R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V',
                        'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'}

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.prot2seq = self._read_fasta()
        self.prot_list = self._load_protein_list()
        self.prot2annot, self.goterms, self.gonames = self._load_annotations()

    def _read_fasta(self) -> Dict[str, str]:
        """Read and validate protein sequences from FASTA file"""
        prot2seq = {}
        opener = gzip.open if str(self.config.sequence_file).endswith('gz') else open

        with opener(self.config.sequence_file, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                prot = self._normalize_protein_id(record.id)

                if self._validate_sequence(seq):
                    prot2seq[prot] = seq

        return prot2seq

    @staticmethod
    def _normalize_protein_id(prot_id: str) -> str:
        """Normalize protein ID to standard format"""
        pdb, chain = prot_id.split('_') if '_' in prot_id else prot_id.split('-')
        return f"{pdb.upper()}-{chain}"

    def _validate_sequence(self, seq: str) -> bool:
        """Validate sequence length and amino acid composition"""
        return (self.config.min_seq_length <= len(seq) <= self.config.max_seq_length and
                not set(seq).difference(self.VALID_AMINO_ACIDS))

    def _load_protein_list(self) -> List[str]:
        """Load list of protein IDs"""
        with open(self.config.protein_list_file) as f:
            return [line.strip() for line in f]

    def process_to_tfrecord(self) -> None:
        """Convert sequences to TFRecord format"""
        shard_size = len(self.prot_list) // self.config.num_shards
        indices = [(i * shard_size, (i + 1) * shard_size)
                  for i in range(self.config.num_shards)]
        indices[-1] = (indices[-1][0], len(self.prot_list))

        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            executor.map(self._process_shard, range(self.config.num_shards))

    def _process_shard(self, shard_idx: int) -> None:
        """Process a single shard of the dataset"""
        tfrecord_path = (self.config.tfrecord_prefix /
                        f"shard_{shard_idx:02d}-of-{self.config.num_shards:02d}.tfrecord")

        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            start_idx, end_idx = self._get_shard_indices(shard_idx)
            for prot_id in self.prot_list[start_idx:end_idx]:
                example = self._create_tf_example(prot_id)
                writer.write(example.SerializeToString())

    def _create_tf_example(self, prot_id: str) -> tf.train.Example:
        """Create TF Example from protein data"""
        sequence = self.prot2seq[prot_id]

        features = {
            'prot_id': self._bytes_feature(prot_id.encode()),
            'seq_1hot': self._float_feature(self._seq2onehot(sequence)),
            'L': self._int64_feature(len(sequence)),
            'ht50_res_embed': self._float_feature(self._get_residue_embedding(prot_id)),
            'ht50_prot_embed': self._float_feature(self._get_protein_embedding(prot_id)),
            **self._create_label_features(prot_id)
        }

        return tf.train.Example(features=tf.train.Features(feature=features))

    def _load_annotations(self) -> Tuple[Dict, Dict, Dict]:
        """Load GO annotations from TSV file"""
        onts = ['molecular_function', 'biological_process', 'cellular_component']
        prot2annot = {}
        goterms = {ont: [] for ont in onts}
        gonames = {ont: [] for ont in onts}

        with open(self.config.annotation_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')

            # Load terms for each ontology
            for ont in onts:
                next(reader, None)  # skip headers
                goterms[ont] = next(reader)
                next(reader, None)  # skip headers
                gonames[ont] = next(reader)

            next(reader, None)  # skip final header
            for row in reader:
                prot, *prot_goterms = row
                prot2annot[prot] = {ont: [] for ont in onts}
                for i, ont in enumerate(onts):
                    goterm_indices = [
                        goterms[ont].index(term)
                        for term in prot_goterms[i].split(',')
                        if term
                    ]
                    labels = np.zeros(len(goterms[ont]), dtype=np.int64)
                    labels[goterm_indices] = 1
                    prot2annot[prot][ont] = labels

        return prot2annot, goterms, gonames

    def _create_label_features(self, prot_id: str) -> Dict:
        """Create GO annotation label features"""
        return {
            'mf_labels': self._int64_feature(self.prot2annot[prot_id]['molecular_function']),
            'bp_labels': self._int64_feature(self.prot2annot[prot_id]['biological_process']),
            'cc_labels': self._int64_feature(self.prot2annot[prot_id]['cellular_component'])
        }

    @staticmethod
    def _int64_feature(value) -> tf.train.Feature:
        """Create int64 feature from value"""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value) -> tf.train.Feature:
        """Create bytes feature from value"""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value) -> tf.train.Feature:
        """Create float feature from value or list of values"""
        if isinstance(value, np.ndarray):
            value = value.flatten().tolist()
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _seq2onehot(self, sequence: str) -> np.ndarray:
        """Convert amino acid sequence to one-hot encoding"""
        # Create mapping of amino acids to indices
        aa_to_idx = {aa: idx for idx, aa in enumerate(sorted(self.VALID_AMINO_ACIDS))}

        # Create one-hot encoding matrix
        onehot = np.zeros((len(sequence), len(self.VALID_AMINO_ACIDS)), dtype=np.float32)
        for i, aa in enumerate(sequence):
            onehot[i, aa_to_idx[aa]] = 1.0
        return onehot

    def _get_shard_indices(self, shard_idx: int) -> Tuple[int, int]:
        """Get start and end indices for a shard"""
        shard_size = len(self.prot_list) // self.config.num_shards
        start_idx = shard_idx * shard_size
        end_idx = start_idx + shard_size if shard_idx < self.config.num_shards - 1 else len(self.prot_list)
        return start_idx, end_idx

    def _get_residue_embedding(self, prot_id: str) -> np.ndarray:
        """Get residue-level embeddings for a protein"""
        with h5py.File(self.config.embeddings_file, 'r') as f:
            return f[f"{prot_id}/residue_embedding"][:]

    def _get_protein_embedding(self, prot_id: str) -> np.ndarray:
        """Get protein-level embedding for a protein"""
        with h5py.File(self.config.embeddings_file, 'r') as f:
            return f[f"{prot_id}/protein_embedding"][:]