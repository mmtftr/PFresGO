import pytest
from pathlib import Path
import tempfile
import numpy as np
from preprocessing.dataset_processor import DatasetConfig, SequenceProcessor

@pytest.fixture
def test_data_dir():
    """Create temporary directory with test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test FASTA file
        with open(tmp_path / "test.fasta", "w") as f:
            f.write(">1ABC-A\nMKLPVRGS\n>2XYZ-B\nACDEFGHIKLMNPQRSTVWY\n")

        # Create test protein list
        with open(tmp_path / "test_proteins.txt", "w") as f:
            f.write("1ABC-A\n2XYZ-B\n")

        # Create test annotation file
        with open(tmp_path / "test_annot.tsv", "w") as f:
            f.write("molecular_function\nGO:0001\nGO name 1\n")
            f.write("biological_process\nGO:0002\nGO name 2\n")
            f.write("cellular_component\nGO:0003\nGO name 3\n")
            f.write("protein\tGO terms\n")
            f.write("1ABC-A\tGO:0001,GO:0002,GO:0003\n")

        yield tmp_path

@pytest.fixture
def config(test_data_dir):
    """Create test configuration"""
    return DatasetConfig(
        annotation_file=test_data_dir / "test_annot.tsv",
        protein_list_file=test_data_dir / "test_proteins.txt",
        sequence_file=test_data_dir / "test.fasta",
        tfrecord_prefix=test_data_dir / "test_tfrecord",
        num_threads=1,
        num_shards=1
    )

def test_sequence_processor_initialization(config):
    """Test if SequenceProcessor initializes correctly"""
    processor = SequenceProcessor(config)
    assert len(processor.prot2seq) == 2
    assert "1ABC-A" in processor.prot2seq
    assert "2XYZ-B" in processor.prot2seq

def test_protein_id_normalization():
    """Test protein ID normalization"""
    assert SequenceProcessor._normalize_protein_id("1abc_A") == "1ABC-A"
    assert SequenceProcessor._normalize_protein_id("1abc-A") == "1ABC-A"

def test_sequence_validation(config):
    """Test sequence validation"""
    processor = SequenceProcessor(config)
    assert processor._validate_sequence("MKLEPVR")
    assert not processor._validate_sequence("MKL1PVR")  # Contains invalid character
    assert not processor._validate_sequence("M")  # Too short

def test_tfrecord_creation(config):
    """Test TFRecord creation"""
    processor = SequenceProcessor(config)
    processor.process_to_tfrecord()

    # Check if TFRecord file was created
    tfrecord_files = list(config.tfrecord_prefix.glob("*.tfrecord"))
    assert len(tfrecord_files) == 1

    # Verify TFRecord content
    dataset = tf.data.TFRecordDataset(str(tfrecord_files[0]))
    example = next(iter(dataset))

    # Parse example and verify contents
    feature_description = {
        'prot_id': tf.io.FixedLenFeature([], tf.string),
        'seq_1hot': tf.io.VarLenFeature(tf.float32),
        'L': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed_features = tf.io.parse_single_example(example, feature_description)
    assert parsed_features['L'].numpy() > 0