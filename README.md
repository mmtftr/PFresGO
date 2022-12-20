# PFresGO: an attention mechanism-based deep-learning approach for protein annotation by integrating gene ontology inter-relationships

PFresGO is an attention-based deep-learning approach that incorporates hierarchical structures in Gene Ontology (GO) graphs and advances in natural language processing algorithms for the func-tional annotation of proteins.

This repository contains script which were used to train the PFresGO model together with the scripts for conducting protein function prediction.

## Dependencies
* The code was developed and tested using python 3.7.
* TensorFlow = 2.4.1


## Scripts
### train_PFresGO.py - this script is to train our model PFresGO. 

If you want to trian PFresGO, run:

`python train_PFresGO.py --num_hidden_layers 1 --ontology 'bp' --model_name 'BP_PFresGO'`

### predict.py - this script is to make protrein function prediction. 

If you want to use PFresGO for prediction, prepare your sequence file in the fasta format, generate protein residual level embedding (follow script fasta-embedding.py), put the .h5 format file into ./Datasets/ and run:

`python predict.py --num_hidden_layers 1 --ontology 'bp' --model_name 'BP_PFresGO' --res_embeddings './Datasets/per_residue_embeddings.h5'` 

### train_autoencoder.py - this script is to generate pretrained autoencoder model. 

If you want to trian the autoencoder, run:

`python train_autoencoder.py --input_dims 1024 --model_name 'Autoencoder_128'`

### fasta-embedding.py - this script is to generate protein residual level embedding. 

The protein residual level embedding is generated by pretrained language model protT5. Before you run this script, you need to install the protT5 package via:

`!pip install torch transformers sentencepiece h5py ` 

then put your own fasta format protein sequences into ./Datasets/ and run:

`fasta-embedding.py --seq_path './Datasets/nrPDB-GO_2019.06.18_sequences.fasta'`

The detailed configuration can refer to https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing#scrollTo=QMoeBQnUCK_E

### label_embedding.py - this script is to generate GO term embedding 

The GO term embedding is generated by pretrained model Anc2Vec. Before you run this script, put your own .obo format GO terms file into ./Datasets/ and install the Anc2Vec package via:

`pip install -U "anc2vec @ git+https://github.com/aedera/anc2vec.git"`

The detailed configuration can refer to https://github.com/sinc-lab/anc2vec

### Datasets - Here you can find the data used to train our method and make prediction.









