import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
import tempfile
import os

from autoencoder import AutoEncoder, AutoEncoderConfig

@pytest.fixture
def config():
    """Basic config fixture for testing"""
    return AutoEncoderConfig(
        input_dim=10,
        hidden_dims=[8, 4],
        batch_size=2,
        num_steps=2,
        learning_rate=0.01
    )

def create_mock_dataset(config):
    print("\nDebug: Creating mock dataset")

    # Create data with explicit batch size
    num_samples = 100
    batch_size = config.batch_size

    # Generate data
    data = tf.random.normal((num_samples, 5, config.input_dim))

    # Create dataset with explicit types and shapes
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.cast(data, tf.float32)
    ).batch(
        batch_size, drop_remainder=True
    ).prefetch(tf.data.AUTOTUNE)

    print(f"Dataset element spec: {dataset.element_spec}")
    return dataset

@pytest.fixture
def mock_dataset(config):
    return create_mock_dataset(config)

def test_autoencoder_initialization(config):
    """Test if AutoEncoder initializes correctly"""
    model = AutoEncoder(config)
    assert model.config == config
    assert model.model is not None
    assert isinstance(model.model, tf.keras.Model)
    assert model.history is None

def test_model_architecture(config):
    """Test if the model architecture is built correctly"""
    model = AutoEncoder(config)

    # Check input shape
    assert model.model.input_shape == (None, None, config.input_dim)

    # Check output shape matches input shape
    assert model.model.output_shape == (None, None, config.input_dim)

    # Check layer names and types
    layer_names = [layer.name for layer in model.model.layers]
    assert 'sequence_input' in layer_names
    assert 'encoder_0' in layer_names
    assert 'encoder_1' in layer_names
    assert 'decoder_0' in layer_names
    assert 'reconstruction' in layer_names

def test_model_training(config, mock_dataset):
    model = AutoEncoder(config)

    # Calculate steps based on dataset size
    total_samples = 100  # Same as in create_mock_dataset
    steps_per_epoch = total_samples // config.batch_size

    history = model.train(
        train_data=mock_dataset,
        valid_data=mock_dataset,
        steps_per_epoch=steps_per_epoch,  # Use calculated steps
        validation_steps=steps_per_epoch   # Same for validation
    )

    assert history is not None

def test_model_saving(config):
    """Test if the model saves correctly"""
    model = AutoEncoder(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test default path
        model.save_model()
        default_path = f'./trained_model_{config.model_name}.h5'
        assert os.path.exists(default_path)
        os.remove(default_path)

        # Test custom path
        custom_path = os.path.join(tmpdir, 'custom_model.h5')
        model.save_model(custom_path)
        assert os.path.exists(custom_path)

def test_early_stopping(config, mock_dataset):
    """Test if early stopping works"""
    config.early_stopping_patience = 1
    model = AutoEncoder(config)

    model.train(
        train_data=mock_dataset,
        valid_data=mock_dataset,
        steps_per_epoch=2,
        validation_steps=2
    )

    # Check if training stopped early
    assert len(model.history['loss']) <= config.num_steps

def test_checkpointing(config, mock_dataset):
    """Test if model checkpointing works"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.h5')
        config.checkpoint_path = checkpoint_path

        model = AutoEncoder(config)
        model.train(
            train_data=mock_dataset,
            valid_data=mock_dataset,
            steps_per_epoch=2,
            validation_steps=2
        )

        assert os.path.exists(checkpoint_path)

def test_invalid_config():
    """Test if invalid configurations raise appropriate errors"""
    with pytest.raises(ValueError):
        AutoEncoderConfig(
            input_dim=-1,
            hidden_dims=[8, 4]
        )

    with pytest.raises(ValueError):
        AutoEncoderConfig(
            input_dim=10,
            hidden_dims=[]
        )

@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_different_activations(activation, config, mock_dataset):
    """Test if model works with different activation functions"""
    config.activation = activation
    model = AutoEncoder(config)

    model.train(
        train_data=mock_dataset,
        valid_data=mock_dataset,
        steps_per_epoch=2,
        validation_steps=2
    )

    assert model.history is not None

def test_prediction(config, mock_dataset):
    """Test if model can make predictions"""
    model = AutoEncoder(config)

    # Get a batch of data
    for batch in mock_dataset:
        predictions = model.model.predict(batch)
        # Check that predictions have same shape as input
        assert predictions.shape == batch.shape
        assert predictions.shape[-1] == config.input_dim
        break
