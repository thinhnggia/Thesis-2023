class Configs:
    # Model configuration
    DROPOUT = 0.5
    CNN_FILTERS = 100
    CNN_KERNEL_SIZE = 5
    LSTM_UNITS = 100
    EMBEDDING_DIM = 50
    MODEL_NAME = "cts_bert_prompt"
    # CHUNK_SIZES = [90, 30, 130,10]
    MODE = "use_bert"
    SHUFFLE_TYPE = None

    # Data configuration
    DATA_PATH = 'src/data/ASAP/final_data'
    FEATURES_PATH = 'src/data/ASAP/hand_crafted_v3.csv'
    READABILITY_PATH = 'src/data/ASAP/allreadability.pickle'
    
    # Training configuration
    EPOCHS = 30
    BATCH_SIZE = 7
    OUTPUT_PATH = 'outputs'
    OPTIMIZER = "rmsprop"
    PRETRAIN = False
    PRETRAIN_BERT = True
