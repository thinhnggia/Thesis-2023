import os
import argparse
import random
# import torch
import wandb

import numpy as np
import tensorflow as tf

from transformers import BertTokenizer
# from torch.optim import RMSprop
# from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
from wandb.keras import WandbCallback
from tensorflow.keras.optimizers import SGD

from src.utils.dataset import parse_dataset
from src.dataset.asap_dataset import get_dataset
from src.config.config import Configs
from src.evaluate.loss import tf_masked_loss_function
from src.evaluate.evaluate import TFEvaluator
from src.models import get_tf_model


def get_optimizer(opt):
    if opt == "rmsprop":
        return opt
    elif opt == "sgd":
        return SGD(lr=1e-3, momentum=0.9)
    else:
        raise NotImplementedError("Opt is not yet supported ")


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# def train_torch(args, config, dataset):
#     """
#     Trainer for pytorch
#     """
#     train_dataset, dev_dataset, test_dataset = dataset["datasets"]

#     # Build model
#     model = get_torch_model(
#         model_name=config.MODEL_NAME,
#         pos_vocab_size=len(dataset["pos_vocab"]),
#         maxnum=dataset["max_sentnum"],
#         maxlen=dataset["max_sentlen"],
#         readability_count=dataset["readability_feature_count"],
#         linguistic_count=dataset["linguistic_feature_count"],
#         config=config,
#         output_dim=dataset["output_dim"]
#     )

#     # Convert to pytorch foramat
#     train_dataset.set_format("torch")
#     dev_dataset.set_format("torch")
#     test_dataset.set_format("torch")

#     train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE)
#     dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=32)
#     test_datalaoder = DataLoader(test_dataset, shuffle=False, batch_size=32)

#     # Build optimizer
#     optimizer = RMSprop(model.parameters(), lr=1e-3)
#     num_training_steps = config.EPOCHS * len(train_dataloader)
#     lr_scheduler = get_scheduler(
#         name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
#     )
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model.to(device)

#     # Evaluator
#     evaluator = Evaluator(
#         test_prompt_id=args.test_prompt_id,
#         dev_dataloader=dev_dataloader,
#         test_dataloader=test_datalaoder,
#         save_path=config.OUTPUT_PATH
#     )

#     # Training loop
#     tbar = tqdm(range(config.EPOCHS), total=config.EPOCHS)
#     for epoch in tbar:
#         tbar.set_description(f"[Epoch {epoch}]")

#         # Trainining
#         model.train()

#         epoch_loss = []
#         mini_tbar = tqdm(train_dataloader, total=len(train_dataloader))
#         for batch in mini_tbar:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(pos=batch["pos"], linguistic=batch["linguistic"], readability=batch["readability"])
#             loss = masked_loss_function(outputs, batch["scores"])
#             loss.backward()
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
            
#             loss = loss.detach().cpu().numpy().item()
#             epoch_loss.append(loss)
#             mini_tbar.set_postfix(loss=loss)

#             del loss
            
#         epoch_loss = np.mean(epoch_loss).item()
#         tbar.set_postfix(epoch_loss=epoch_loss)
        
#         # Evaluation on each epoch
#         evaluator.evaluate(model, epoch, device)

#     evaluator.print_final_info()


def constrain_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=10240)])
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def train_tf(args, config, dataset):
    """
    Trainer for Tensorflow
    """
    output_path = os.path.join(config.OUTPUT_PATH, config.MODEL_NAME)

    # Set wandb config
    run_name = f"prompt_{args.test_prompt_id}"
    wandb.init(
        project="Thesis",
        config={
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "optimizer": config.OPTIMIZER
        },
        group=f"{config.MODEL_NAME}_custom_split_max_pool",
        name=run_name
    )

    # Set to use bert or not
    mode = config.MODE

    train_dataset, dev_dataset, test_dataset = dataset["datasets"]
    # train_dataset_tf, dev_dataset_tf, test_dataset_tf = dataset["tf_datasets"]  # Uncomment if use custom

    # Build model
    model = get_tf_model(
        args,
        model_name=config.MODEL_NAME,
        freeze=True,
        pos_vocab_size=len(dataset["pos_vocab"]),
        maxnum=dataset["max_sentnum"],
        maxlen=dataset["max_sentlen"],
        readability_feature_count=dataset["readability_feature_count"],
        linguistic_feature_count=dataset["linguistic_feature_count"],
        config=config,
        output_dim=dataset["output_dim"],
        pretrain_name=config.BASE_MODEL_NAME
    )
    
    # # Uncomment for fast model debug
    # model = get_tf_model(
    #     args,
    #     model_name=config.MODEL_NAME,
    #     freeze=True,
    #     pos_vocab_size=36,
    #     maxnum=97,
    #     maxlen=50,
    #     readability_feature_count=35,
    #     linguistic_feature_count=51,
    #     config=config,
    #     output_dim=9,
    #     pretrain_name=config.BASE_MODEL_NAME
    # )
    # import pdb; pdb.set_trace()

    if config.PRETRAIN:
        pretrained_weights = os.path.join(output_path, f"current_model_prompt_{args.test_prompt_id}.h5")
        # pretrained_weights = os.path.join(output_path, f"best_model_prompt_{args.test_prompt_id}.h5")
        print("Reload model weights: ", pretrained_weights)
        model.load_weights(pretrained_weights)

    # Process dataset
    train_inputs, _, Y_train = parse_dataset(train_dataset, mode) # Uncomment if use not custom
    parsed_dev_dataset = parse_dataset(dev_dataset, mode)
    parsed_test_dataset = parse_dataset(test_dataset, mode)

    # train_dataset_tf = train_dataset_tf.to_tf_dataset(
    #     columns=["input_1", "input_2", "input_3", "input_4", "input_5", "input_6"],
    #     label_cols=["scores"],
    #     batch_size=config.BATCH_SIZE,
    #     shuffle=True
    # )

    # Evaluator
    best_model_name = f"best_model_prompt_{args.test_prompt_id}.h5"
    log_file = f"logs/{config.MODEL_NAME}_{run_name}.txt"
    evaluator = TFEvaluator(
        args.test_prompt_id,
        parsed_dev_dataset,
        parsed_test_dataset,
        save_path=output_path,
        model_name=best_model_name,
        log_file=log_file,
        mode=mode
    )

    tbar = tqdm(range(config.EPOCHS), total=config.EPOCHS)
    print(f"Start training for target prompt: {args.test_prompt_id}")

    # Not custom model ------------------------------------------------------------------------
    # Compile model
    if mode == "prompt_tuning":
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        loss_func = tf_masked_loss_function
    model.compile(loss=loss_func, optimizer=get_optimizer(config.OPTIMIZER))

    for epoch in tbar:
        tbar.set_description(f"[Epoch {epoch + 1}]")
        model.fit(train_inputs, Y_train, batch_size=config.BATCH_SIZE, epochs=1, verbose=1, shuffle=True, callbacks=[WandbCallback()])
        evaluator.evaluate(model, epoch + 1, loss_func=loss_func)
    
    # # Custom model ------------------------------------------------------------------------
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    # for epoch in tbar:
    #     tbar.set_description(f"[Epoch {epoch + 1}]")
    #     batch_tbar = tqdm(train_dataset_tf, total=len(train_dataset_tf))
    #     for step, (x_batch_train, y_batch_train) in enumerate(batch_tbar):
    #         with tf.GradientTape() as tape:
    #             outputs = model(x_batch_train, training=True)
    #             loss_value = tf_masked_loss_function(y_batch_train, outputs)

    #         grads = tape.gradient(loss_value, model.trainable_weights)
    #         optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
    #     wandb.log({
    #         "train_loss": loss_value.numpy(),
    #     })
        
    #     evaluator.evaluate(model, epoch + 1)
    
    if not (mode == "prompt_tuning"):
        evaluator.print_final_info()
        evaluator.save_final_info()
    
    wandb.save(log_file)
    wandb.finish()


def main():
    # Init argument
    parser = argparse.ArgumentParser(description="PAES_attributes model")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--mode', type=str, default="tensorflow", help="Framework for training")
    args = parser.parse_args()

    # Seed for random init
    set_seed(args.seed)

    # Init config
    config = Configs()

    # Init trainer
    if args.mode == "tensorflow":
        constrain_gpu()

        bert_tokenizer = None
        if config.MODE == "use_bert" or config.MODE == "prompt_tuning" or config.MODE == "use_custom":
            bert_tokenizer = BertTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        
        # Build dataset
        dataset = get_dataset(config=config, args=args, bert_tokenizer=bert_tokenizer)
        # dataset = None

        # Training
        train_tf(args, config, dataset)
    # elif args.mode == 'pytorch':
    #     train_torch(args, config, dataset)
    # else:
    #     raise ValueError(f"Not implemented framework: {args.mode}")


if __name__ == '__main__':
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
    main()
