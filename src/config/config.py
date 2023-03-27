class Configs:
    DROPOUT = 0.5
    CNN_FILTERS = 100
    CNN_KERNEL_SIZE = 5
    LSTM_UNITS = 100
    EMBEDDING_DIM = 50
    PRETRAINED_EMBEDDING = True
    EMBEDDING_PATH = 'embeddings/glove.6B.50d.txt'
    VOCAB_SIZE = 4000
    DATA_PATH = 'src/data/ASAP/final_data'
    FEATURES_PATH = 'src/data/ASAP/hand_crafted_v3.csv'
    READABILITY_PATH = 'src/data/ASAP/allreadability.pickle'
    EPOCHS = 50
    BATCH_SIZE = 10
    OUTPUT_PATH = 'outputs/'
