import os
import argparse
import random
import numpy as np
import torch

from torch.optim import RMSprop
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm

from src.dataset.asap_dataset import get_dataset
from src.config.config import Configs
from src.evaluate.loss import masked_loss_function
from src.models.cts import CTS
from src.evaluate.evaluate import Evaluator


def main():
    # Init argument
    parser = argparse.ArgumentParser(description="PAES_attributes model")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--model_name', type=str, help='name of model')
    parser.add_argument('--mode', type=str, default="native", help="Mode of training pytorch")
    args = parser.parse_args()

    # Seed for random init
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Init config
    config = Configs()
    
    # Build dataset
    dataset = get_dataset(config=config, args=args)
    train_dataset, dev_dataset, test_dataset = dataset["datasets"]

    # Build model
    model = CTS(
        pos_vocab_size=len(dataset["pos_vocab"]),
        maxum=dataset["max_sentnum"],
        maxlen=dataset["max_sentlen"],
        readability_count=dataset["readability_feature_count"],
        linguistic_count=dataset["linguistic_feature_count"],
        config=config,
        output_dim=dataset["output_dim"]
    )

    if args.mode == "native":
        # Convert to pytorch foramat
        train_dataset.set_format("torch")
        dev_dataset.set_format("torch")
        test_dataset.set_format("torch")

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=32)
        test_datalaoder = DataLoader(test_dataset, shuffle=False, batch_size=32)

        # Build optimizer
        optimizer = RMSprop(model.parameters(), lr=1e-3)
        num_training_steps = config.EPOCHS * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Evaluator
        evaluator = Evaluator(
            test_prompt_id=args.test_prompt_id,
            dev_dataloader=dev_dataloader,
            test_dataloader=test_datalaoder,
            save_path=config.OUTPUT_PATH
        )

        # Training loop
        tbar = tqdm(range(config.EPOCHS), total=config.EPOCHS)
        for epoch in tbar:
            tbar.set_description(f"[Epoch {epoch}]")

            # Trainining
            model.train()

            epoch_loss = []
            mini_tbar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in mini_tbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(pos=batch["pos"], linguistic=batch["linguistic"], readability=batch["readability"])
                loss = masked_loss_function(outputs, batch["scores"])
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                loss = loss.detach().cpu().numpy().item()
                epoch_loss.append(loss)
                mini_tbar.set_postfix(loss=loss)

                del loss
                
            epoch_loss = np.mean(epoch_loss).item()
            tbar.set_postfix(epoch_loss=epoch_loss)
            
            # Evaluation on each epoch
            evaluator.evaluate(model, epoch, device)

        evaluator.print_final_info()
    else:
        raise ValueError("Not implemented")

if __name__ == '__main__':
    main()
