import argparse
import tensorflow as tf
from autoencoder import AutoEncoder, AutoEncoderConfig

def create_dataset(tfrecord_pattern: str, config: AutoEncoderConfig) -> tf.data.Dataset:
    """Create a dataset from TFRecord files"""
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_pattern))
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-idim', '--input_dims', type=int, default=1024, help="Input dimension")
    parser.add_argument('-hdim', '--hidden_dims', type=int, nargs='+', default=[1024,256,128], help="Dimensions of middle layers")
    parser.add_argument('-lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs to train")
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--model_name', type=str, default='Autoencoder_128', help="Name of the autoencoder model")
    parser.add_argument('--train_tfrecord_fn', type=str, default='./Datasets/TFRecords_sequences/PDB_GO_train', help='Train tfrecords')
    parser.add_argument('--valid_tfrecord_fn', type=str, default='./Datasets/TFRecords_sequences/PDB_GO_valid', help='Valid tfrecords')
    parser.add_argument('--checkpoint_path', type=str, help='Path to save model checkpoints')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()

    # Create config
    config = AutoEncoderConfig(
        input_dim=args.input_dims,
        hidden_dims=args.hidden_dims,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_steps=args.epochs,
        model_name=args.model_name,
        early_stopping_patience=args.patience,
        checkpoint_path=args.checkpoint_path
    )

    # Create datasets
    train_dataset = create_dataset(args.train_tfrecord_fn + '*', config)
    valid_dataset = create_dataset(args.valid_tfrecord_fn + '*', config)

    # Calculate steps per epoch (assuming you know the total number of samples)
    # You might need to adjust these values based on your actual dataset size
    steps_per_epoch = 1000 // config.batch_size  # Replace 1000 with actual training samples
    validation_steps = 200 // config.batch_size   # Replace 200 with actual validation samples

    # Create and train model
    model = AutoEncoder(config)
    history = model.train(
        train_data=train_dataset,
        valid_data=valid_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # Save model
    model.save_model()
    print("Training finished!")






