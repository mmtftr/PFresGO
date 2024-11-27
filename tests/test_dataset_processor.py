import pytest
from pathlib import Path
import tempfile
import numpy as np
import tensorflow as tf
from preprocessing.dataset_processor import DatasetConfig, SequenceProcessor

@pytest.fixture
def test_data_dir(tmp_path):
    """Create test data directory with necessary files"""
    # Create test FASTA file
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(
        ">1ABC-A\nMKLPVRGS\n"
        ">2XYZ-B\nACDEFGHIKLMNPQRSTVWY\n"
    )

    # Create test protein list
    prot_list_file = tmp_path / "test_proteins.txt"
    prot_list_file.write_text(
        "1ABC-A\n"
        "2XYZ-B\n"
    )

    # Create test annotation file
    annot_file = tmp_path / "test_annot.tsv"
    annot_file.write_text(
        "molecular_function\n"
        "GO:0001\n"
        "GO name 1\n"
        "biological_process\n"
        "GO:0002\n"
        "GO name 2\n"
        "cellular_component\n"
        "GO:0003\n"
        "GO name 3\n"
        "protein\tGO terms\n"
        "1ABC-A\tGO:0001,,\n"
        "2XYZ-B\t,GO:0002,GO:0003\n"
    )

    # Create mock embeddings file
    embeddings_file = tmp_path / "embeddings.h5"
    with h5py.File(embeddings_file, 'w') as f:
        f.create_dataset("1ABC-A nrPDB", data=np.random.rand(8, 1024))
        f.create_dataset("2XYZ-B nrPDB", data=np.random.rand(19, 1024))

    return tmp_path

@pytest.fixture
def config(test_data_dir):
    """Create test configuration"""
    return DatasetConfig(
        annotation_file=test_data_dir / "test_annot.tsv",
        protein_list_file=test_data_dir / "test_proteins.txt",
        sequence_file=test_data_dir / "test.fasta",
        tfrecord_prefix=test_data_dir / "test_tfrecord",
        embeddings_file=test_data_dir / "embeddings.h5",
        num_threads=1,
        num_shards=1,
        min_seq_length=5,  # Smaller for testing
        max_seq_length=25  # Smaller for testing
    )

def test_sequence_processor_initialization(config):
    """Test if SequenceProcessor initializes correctly"""
    processor = SequenceProcessor(config)
    assert len(processor.prot2seq) == 2
    assert "1ABC-A" in processor.prot2seq
    assert "2XYZ-B" in processor.prot2seq
    assert processor.prot2seq["1ABC-A"] == "MKLPVRGS"

def test_sequence_validation(config):
    """Test sequence validation"""
    processor = SequenceProcessor(config)
    # Valid sequence
    assert processor._validate_sequence("MKLEPVR")
    # Invalid sequences
    assert not processor._validate_sequence("MKL1PVR")  # Contains invalid character
    assert not processor._validate_sequence("M")  # Too short
    assert not processor._validate_sequence("A" * (config.max_seq_length + 1))  # Too long

def test_tfrecord_creation(config):
    """Test TFRecord creation"""
    processor = SequenceProcessor(config)
    processor.process_to_tfrecord()

    # Check if TFRecord file was created
    tfrecord_files = list(config.tfrecord_prefix.parent.glob("test_tfrecord*.tfrecord"))
    assert len(tfrecord_files) == 1

    # Read and verify TFRecord content
    feature_description = {
        'prot_id': tf.io.FixedLenFeature([], tf.string),
        'seq_1hot': tf.io.VarLenFeature(tf.float32),
        'L': tf.io.FixedLenFeature([], tf.int64),
        'ht50_res_embed': tf.io.VarLenFeature(tf.float32),
        'mf_labels': tf.io.FixedLenFeature([1], tf.int64),
        'bp_labels': tf.io.FixedLenFeature([1], tf.int64),
        'cc_labels': tf.io.FixedLenFeature([1], tf.int64),
    }

    dataset = tf.data.TFRecordDataset(str(tfrecord_files[0]))
    for raw_record in dataset.take(1):
        example = tf.io.parse_single_example(raw_record, feature_description)
        assert example['L'].numpy() > 0
        assert example['prot_id'].numpy().decode('utf-8') in ["1ABC-A", "2XYZ-B"]

def test_protein_id_normalization():
    """Test protein ID normalization"""
    assert SequenceProcessor._normalize_protein_id("1abc_A") == "1ABC-A"
    assert SequenceProcessor._normalize_protein_id("1abc-A") == "1ABC-A"
    assert SequenceProcessor._normalize_protein_id("2xyz_B") == "2XYZ-B"

def test_invalid_config():
    """Test configuration validation"""
    with pytest.raises(FileNotFoundError):
        DatasetConfig(
            annotation_file=Path("nonexistent.tsv"),
            protein_list_file=Path("nonexistent.txt"),
            sequence_file=Path("nonexistent.fasta"),
            tfrecord_prefix=Path("output"),
            embeddings_file=Path("nonexistent.h5")
        )

def test_sequence_processing(config):
    """Test sequence to one-hot encoding"""
    processor = SequenceProcessor(config)
    seq = "MKLP"
    one_hot = processor._seq2onehot(seq)
    assert one_hot.shape == (4, 26)  # 4 residues, 26 possible amino acids
    assert np.sum(one_hot) == 4  # One-hot encoding should have exactly one 1 per row