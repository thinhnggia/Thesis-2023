import os
import torch
import numpy as np
import tensorflow as tf
import wandb

from src.evaluate.metric import kappa
from src.utils.metric import separate_and_rescale_attributes_for_scoring, separate_attributes_for_scoring
from src.evaluate.loss import tf_masked_loss_function

class TFEvaluator():
    def __init__(self, test_prompt_id, dev_dataset, test_dataset, save_path="checkpoints", model_name="best_model.h5", log_file="logs/log.txt"):
        self.test_prompt_id = test_prompt_id

        # Create model path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model_path = os.path.join(save_path, model_name)

        # Log model path
        log_path = os.path.dirname(log_file)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_file = log_file
        
        if os.path.isfile(log_file):
            os.remove(log_file)

        dev_features_list, X_dev_prompt_ids, Y_dev = dev_dataset
        test_features_list, X_test_prompt_ids, Y_test = test_dataset

        self.dev_features_list = dev_features_list
        self.test_features_list = test_features_list

        self.X_dev_prompt_ids, self.X_test_prompt_ids = X_dev_prompt_ids, X_test_prompt_ids
        self.Y_dev, self.Y_test = Y_dev, Y_test
        self.Y_dev_upscale = Y_dev * 100
        self.Y_dev_org = separate_attributes_for_scoring(self.Y_dev_upscale, self.X_dev_prompt_ids)
        self.Y_test_org = separate_and_rescale_attributes_for_scoring(Y_test, self.X_test_prompt_ids)
        self.best_dev_kappa_mean = -1
        self.best_test_kappa_mean = -1
        self.best_dev_kappa_set = {}
        self.best_test_kappa_set = {}
        self.best_dev_loss = np.inf

    @staticmethod
    def calc_kappa(pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score

    def evaluate(self, model, epoch, print_info=True, save_info=True):
        self.current_epoch = epoch

        dev_pred = model.predict(self.dev_features_list, batch_size=2)
        test_pred = model.predict(self.test_features_list, batch_size=2)
        dev_loss = tf_masked_loss_function(self.Y_dev, dev_pred)

        dev_pred_int = dev_pred * 100
        dev_pred_dict = separate_attributes_for_scoring(dev_pred_int, self.X_dev_prompt_ids)

        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred, self.X_test_prompt_ids)

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], self.Y_dev_org[key]) for key in
                        dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], self.Y_test_org[key]) for key in
                         test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        wandb.log({
            "dev_loss": dev_loss.numpy(),
            "dev_kappa_mean": self.dev_kappa_mean,
            "test_kappa_mean": self.test_kappa_mean
        })

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            self.best_dev_loss = dev_loss
            model.save_weights(self.model_path, save_format="h5")

        if print_info:
            self.print_info()

        if save_info:
            self.save_info()

        # Free memory
        del dev_pred
        del test_pred
        del dev_loss

    def print_info(self):
        print('CURRENT EPOCH: {}'.format(self.current_epoch))
        print('[DEV] AVG QWK: {}'.format(round(self.dev_kappa_mean, 3)))
        for att in self.kappa_dev.keys():
            print('[DEV] {} QWK: {}'.format(att, round(self.kappa_dev[att], 3)))
        print(
            '------------------------')
        print('[TEST] AVG QWK: {}'.format(round(self.test_kappa_mean, 3)))
        for att in self.kappa_test.keys():
            print('[TEST] {} QWK: {}'.format(att, round(self.kappa_test[att], 3)))
        print(
            '------------------------')
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print(
            '--------------------------------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print(
            '--------------------------------------------------------------------------------------------------------------------------')

    def save_info(self):
        with open(self.log_file, "a") as f:
            f.write('CURRENT EPOCH: {}\n'.format(self.current_epoch))
            f.write('[DEV] AVG QWK: {}\n'.format(round(self.dev_kappa_mean, 3)))
            for att in self.kappa_dev.keys():
                f.write('[DEV] {} QWK: {}\n'.format(att, round(self.kappa_dev[att], 3)))
            f.write(
                '------------------------\n')
            f.write('[TEST] AVG QWK: {}\n'.format(round(self.test_kappa_mean, 3)))
            for att in self.kappa_test.keys():
                f.write('[TEST] {} QWK: {}\n'.format(att, round(self.kappa_test[att], 3)))
            f.write(
                '------------------------')
            f.write('[BEST TEST] AVG QWK: {}, {{epoch}}: {}\n'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
            for att in self.best_test_kappa_set.keys():
                f.write('[BEST TEST] {} QWK: {}\n'.format(att, round(self.best_test_kappa_set[att], 3)))
            f.write(
                '--------------------------------------------------------------------------------------------------------------------------\n')
    
    def save_final_info(self):
        with open(self.log_file, "a") as f:
            f.write('[BEST TEST] AVG QWK: {}, {{epoch}}: {}\n'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
            for att in self.best_test_kappa_set.keys():
                f.write('[BEST TEST] {} QWK: {}\n'.format(att, round(self.best_test_kappa_set[att], 3)))
            f.write(
                '--------------------------------------------------------------------------------------------------------------------------\n')


class Evaluator():
    """
    Class for evaluation Pytorch
    """
    def __init__(self, test_prompt_id, dev_dataloader, test_dataloader, save_path="checkpoints", model_name="best_model.pth"):
        self.test_prompt_id = test_prompt_id
        self.model_name = model_name
        self.save_path = save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model_path = os.path.join(save_path, model_name)

        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader

        self.X_dev_prompt_ids, self.Y_dev = self.process_dataloader(dev_dataloader)
        self.X_test_prompt_ids, self.Y_test = self.process_dataloader(test_dataloader)
        self.Y_dev_upscale = self.Y_dev * 100
        self.Y_dev_org = separate_attributes_for_scoring(self.Y_dev_upscale, self.X_dev_prompt_ids.tolist())
        self.Y_test_org = separate_and_rescale_attributes_for_scoring(self.Y_test, self.X_test_prompt_ids.tolist())

        self.best_dev_kappa_mean = -1
        self.best_test_kappa_mean = -1
        self.best_dev_kappa_set = {}
        self.best_test_kappa_set = {}

    def process_dataloader(self, dataloader):
        prompt_ids = []
        labels = []
        for batch in dataloader:
            prompt_ids.append(batch['prompt_ids'])
            labels.append(batch['scores'])
        
        prompt_ids = torch.cat(prompt_ids, dim=0).detach().numpy()
        labels = torch.cat(labels, dim=0).detach().numpy()
        return prompt_ids, labels
    
    def process_preds(self, model, dataloader, device):
        final_preds = []
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if k not in set(["scores", "prompt_ids"])}
                preds = model(**batch)
                final_preds.append(preds)
        return torch.cat(final_preds, dim=0).detach().cpu().numpy()

    @staticmethod
    def calc_kappa(pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score

    def evaluate(self, model, epoch, device, print_info=True, early_stopping=-1, dev=1e-4,
                 save_model=True):
        self.current_epoch = epoch

        # Get prediction
        dev_pred = self.process_preds(model, self.dev_dataloader, device)
        test_pred = self.process_preds(model, self.test_dataloader, device)

        dev_pred_int = dev_pred * 100
        dev_pred_dict = separate_attributes_for_scoring(dev_pred_int, self.X_dev_prompt_ids)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred, self.X_test_prompt_ids)

        self.kappa_dev = {key: self.calc_kappa(dev_pred_dict[key], self.Y_dev_org[key]) for key in
                        dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(test_pred_dict[key], self.Y_test_org[key]) for key in
                         test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch
            if save_model:
                torch.save(model.state_dict(), self.model_path)
        if print_info:
            self.print_info()

    def print_info(self):
        print('CURRENT EPOCH: {}'.format(self.current_epoch))
        print('[DEV] AVG QWK: {}'.format(round(self.dev_kappa_mean, 3)))
        for att in self.kappa_dev.keys():
            print('[DEV] {} QWK: {}'.format(att, round(self.kappa_dev[att], 3)))
        print('------------------------')
        print('[TEST] AVG QWK: {}'.format(round(self.test_kappa_mean, 3)))
        for att in self.kappa_test.keys():
            print('[TEST] {} QWK: {}'.format(att, round(self.kappa_test[att], 3)))
        print('------------------------')
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print('--------------------------------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(att, round(self.best_test_kappa_set[att], 3)))
        print('--------------------------------------------------------------------------------------------------------------------------')