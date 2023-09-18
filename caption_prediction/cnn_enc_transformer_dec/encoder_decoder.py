# Imports
import numpy as np
from typing import Optional

# PyTorch Imports
import torch
from torch import nn
from torch.distributions import Categorical
import torchxrayvision as xrv

# PyTorch Lightning Imports
import lightning as L

# Transformers Imports
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import evaluate

# Sklearn Imports
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Project Imports
from utils import get_scheduler



# Class: CNNTransformerCaptioner
class CNNTransformerCaptioner(L.LightningModule):
    def __init__(
        self,
        decoder="distilgpt2",
        freeze_encoder=False,
        clf=None,
        clf_weight=0.1,
        tokenizer=None,
        max_length=100,
        lr=2e-5,
        min_lr=1e-7,
        train_steps=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer

        self.decoder = AutoModelForCausalLM.from_pretrained(decoder, is_decoder=True, add_cross_attention=True)
        self.configure_decoder(max_length)

        self.encoder = xrv.models.DenseNet(weights="densenet121-res224-all").features
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Project encoder output to decoder hidden_size
        self.enc_to_dec_proj = nn.Linear(1024, self.decoder.config.hidden_size, bias=False)

        if clf is not None:
            print(f"Loaded classifier from {clf}")
            self.classifier =  AutoModelForSequenceClassification.from_pretrained(clf)
            self.clf_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.clf_weight = clf_weight
            for p in self.classifier.parameters():
                p.requires_grad = False

        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")          

    def configure_decoder(self, max_length):
        # Update vocab size
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        # Set special tokens
        self.decoder.config.bos_token_id = self.tokenizer.bos_token_id
        self.decoder.config.eos_token_id = self.tokenizer.eos_token_id
        self.decoder.config.pad_token_id = self.tokenizer.pad_token_id

        # Parameters for generation
        self.decoder.config.max_length = max_length
        self.decoder.config.early_stopping = True

    def get_encoder_features(self, pixel_values):
        img_features = self.encoder(pixel_values)
        # reshape from (bs, 1024, w, h) to (bs, 1024, w*h)
        dims = img_features.shape
        img_features = torch.reshape(img_features, (dims[0], dims[1], dims[2]*dims[3]))
        img_features = torch.permute(img_features, (0, 2, 1))
        img_features_proj = self.enc_to_dec_proj(img_features)

        return img_features_proj
    
    def sample(self, logits):
        dist = Categorical(logits=logits)
        sample = dist.sample()
        return sample, dist.log_prob(sample)
    
    def prepare_inputs_for_generation(
        self, input_ids, past=None, use_cache=None, encoder_hidden_states=None, encoder_attention_mask=None
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past, use_cache=use_cache)
        
        input_dict = {
            "attention_mask": decoder_inputs['attention_mask'],
            "input_ids": decoder_inputs["input_ids"],
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict
    
    def greedy(
        self,
        input_ids: torch.LongTensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        inference: Optional[bool] = True,
        **model_kwargs,
    ):

        # init values
        max_length = max_length if max_length is not None else self.decoder.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.decoder.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.decoder.config.eos_token_id
                    
        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        log_probabilities = []
        
        while True:

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.decoder(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True,
            )

            next_tokens_scores = outputs.logits[:, -1, :]

            if not inference:
                next_tokens, next_tokens_logprobs = self.sample(next_tokens_scores)
            else:
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError('If `eos_token_id` is defined, make sure that `pad_token_id` is defined.')
                
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                if not inference: next_tokens_logprobs = next_tokens_logprobs * unfinished_sequences

            if not inference: log_probabilities.append(next_tokens_logprobs.unsqueeze(1))

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens.detach()[:, None]], dim=1)
            model_kwargs['past'] = outputs.past_key_values

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())


            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or input_ids.shape[-1] >= max_length:
                break
        
        if not inference: log_probabilities = torch.cat(log_probabilities, 1).squeeze()
        return input_ids, log_probabilities
    
    def forward(self, pixel_values, input_ids, attention_mask, labels, clf_labels=None):
        bs = pixel_values.shape[0]
        img_features_proj = self.get_encoder_features(pixel_values)
        out = self.decoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels, encoder_hidden_states=img_features_proj)
        text_loss = out["loss"]

        if hasattr(self, "classifier"):
            if clf_labels is None:
                raise ValueError('clf_labels are needed for classification loss')           
            
            # prepare decoder_input_ids (just BOS tokens to start autoregressive generation)
            decoder_input_ids = torch.ones((bs, 1), dtype=torch.long, device=pixel_values.device) * self.decoder.config.bos_token_id
            gen, log_probs = self.greedy(decoder_input_ids, encoder_hidden_states=img_features_proj, inference=False)
            gen_detok = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            
            gen_retok = self.clf_tokenizer(gen_detok, max_length=self.decoder.config.max_length, padding="longest", truncation=True, return_tensors="pt").to(pixel_values.device)
            clf_out = self.classifier(**gen_retok, labels=clf_labels, return_dict=True)
            clf_reward = 1.0 / (clf_out["loss"] + 1e-8)
            clf_loss = -(log_probs.sum(dim=-1) * clf_reward).sum()
            clf_loss /= bs
            
            return text_loss, clf_loss, gen_detok, clf_out["logits"]
        
        return text_loss
    
    def generate(self, pixel_values):
        img_features_proj = self.get_encoder_features(pixel_values)
        decoder_input_ids = torch.ones((pixel_values.shape[0], 1), dtype=torch.long, device=pixel_values.device) * self.decoder.config.bos_token_id
        gen, _ = self.greedy(decoder_input_ids, encoder_hidden_states=img_features_proj, inference=True)
        gen_detok = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return gen_detok

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = get_scheduler(
            optimizer, self.hparams.train_steps, self.hparams.lr, self.hparams.min_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def compute_clf_metrics(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        return metrics

    def compute_text_metrics(self, preds_str, labels):        
        labels[labels == -100] = self.tokenizer.pad_token_id
        labels_str = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        
        bert_score = self.bertscore.compute(predictions=preds_str, references=labels_str, model_type="microsoft/deberta-xlarge-mnli")
        rouge1_score_f1 = self.rouge.compute(predictions=preds_str, references=labels_str, use_aggregator=False, use_stemmer=False)
        
        return {"bert_score": sum(bert_score["f1"]) / len(bert_score["f1"]), "rouge_score": sum(rouge1_score_f1["rouge1"]) / len(rouge1_score_f1["rouge1"])}

    def training_step(self, batch, batch_idx):
        if hasattr(self, "classifier"):
            text_loss, clf_loss, _, clf_preds = self.forward(**batch)
            metrics = self.compute_clf_metrics(clf_preds.cpu().numpy(), batch["clf_labels"].cpu().numpy())
            for k, v in metrics.items():
                self.log(f"train_{k}", v)
            loss = text_loss + self.clf_weight * clf_loss
            self.log("train_text_loss", text_loss)
            self.log("train_clf_loss", clf_loss)
        else:
            loss = self.forward(**batch)

        self.log("train_loss",loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = {}
        if hasattr(self, "classifier"):
            text_loss, clf_loss, gen, clf_preds = self.forward(**batch)
            clf_metrics = self.compute_clf_metrics(clf_preds.cpu().numpy(), batch["clf_labels"].cpu().numpy())
            metrics.update(clf_metrics)
            for k, v in metrics.items():
                self.log(f"val_{k}", v)
            loss = text_loss + self.clf_weight * clf_loss
            self.log("val_text_loss", text_loss)
            self.log("val_clf_loss", clf_loss)
        else:
            loss = self.forward(**batch)
            gen = self.generate(batch["pixel_values"])

        self.log("val_loss", loss)

        text_metrics = self.compute_text_metrics(gen, batch["labels"])
        metrics.update(text_metrics)
        for k,v in metrics.items():
            self.log(f"val_{k}", v)
        
        return loss

    def test_step(self, batch, batch_idx):
        return self.generate(batch["pixel_values"])
