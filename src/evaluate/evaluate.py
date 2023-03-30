import os
import torch
import numpy as np

from src.evaluate.metric import kappa
from src.utils.metric import separate_and_rescale_attributes_for_scoring, separate_attributes_for_scoring


class Evaluator():
    """
    Class for evaluation
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