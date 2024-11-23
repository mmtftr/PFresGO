import tensorflow as tf
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AutoEncoderConfig:
    """Configuration for AutoEncoder model"""
    input_dim: int
    hidden_dims: List[int]
    learning_rate: float = 0.001
    batch_size: int = 32
    num_steps: int = 100
    model_name: str = 'autoencoder'
    activation: str = 'relu'
    optimizer_beta1: float = 0.95
    optimizer_beta2: float = 0.95
    early_stopping_patience: int = 5
    checkpoint_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")

        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("all hidden dimensions must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")

        if not isinstance(self.model_name, str):
            raise ValueError("model_name must be a string")

        if self.activation not in ['relu', 'tanh', 'sigmoid']:
            raise ValueError("activation must be one of: relu, tanh, sigmoid")

class AutoEncoder:
    """
    AutoEncoder implementation using TF 2.x practices

    Args:
        config: AutoEncoderConfig containing model parameters
    """
    def __init__(self, config: AutoEncoderConfig):
        self.config = config
        self.model = self._build_model()
        self.history = None

    def _build_model(self) -> tf.keras.Model:
        """
        Builds the autoencoder architecture using Keras Functional API
        """
        print("Debug: Starting model build")
        print(f"Input dim: {self.config.input_dim}")
        print(f"Hidden dims: {self.config.hidden_dims}")

        # Input layer with explicit name
        input_seq = tf.keras.layers.Input(shape=(5, self.config.input_dim), name='sequence_input')
        print(f"Input shape: {input_seq.shape}")

        # Encoder
        x = input_seq
        for i, dim in enumerate(self.config.hidden_dims):
            x = tf.keras.layers.Dense(
                dim,
                activation=self.config.activation,
                name=f'encoder_{i}'
            )(x)
            print(f"Encoder layer {i} output shape: {x.shape}")

        # Decoder
        for i, dim in enumerate(self.config.hidden_dims[::-1]):
            x = tf.keras.layers.Dense(
                dim,
                activation=self.config.activation,
                name=f'decoder_{i}'
            )(x)
            print(f"Decoder layer {i} output shape: {x.shape}")

        # Final reconstruction
        outputs = tf.keras.layers.Dense(
            self.config.input_dim,
            activation=None,
            name='reconstruction'
        )(x)
        print(f"Final output shape: {outputs.shape}")

        model = tf.keras.Model(inputs=input_seq, outputs=outputs)

        # Use the custom optimizer instance
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            beta_1=self.config.optimizer_beta1,
            beta_2=self.config.optimizer_beta2
        )

        model.compile(
            optimizer=optimizer,  # Use the optimizer instance
            loss="mse",
            metrics=['mae']
        )

        print("\nDebug: Model summary:")
        model.summary()

        return model

    def train(self, train_data, valid_data, steps_per_epoch, validation_steps):
        """
        Train the model with the given data
        """
        print("\nDebug: Starting training")

        # Ensure datasets are properly batched and repeated
        train_data = train_data.repeat()
        valid_data = valid_data.repeat()

        # Add more debugging
        print("Training data spec:", train_data.element_spec)

        callbacks = self._get_callbacks()

        try:
            history = self.model.fit(
                train_data,
                validation_data=valid_data,
                epochs=self.config.num_steps,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            return history
        except Exception as e:
            print(f"\nDebug: Error during training: {str(e)}")
            print(f"Error type: {type(e)}")
            raise

    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save the model to disk

        Args:
            path: Optional custom path to save the model
        """
        save_path = path or f'./trained_model_{self.config.model_name}.h5'
        self.model.save(save_path)

    def _get_callbacks(self) -> list:
        """
        Creates a list of callbacks for model training
        """
        callbacks = []

        # Add early stopping if patience is specified
        if self.config.early_stopping_patience:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        # Add model checkpoint if path is specified
        if self.config.checkpoint_path:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            )
            callbacks.append(checkpoint)

        return callbacks




