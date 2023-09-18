# Imports
import os
import random
import argparse
import datetime
import shutil
import numpy as np

# PyTorch Imports
import torch

# Transformers Imports
from transformers import AutoImageProcessor, AutoTokenizer, EarlyStoppingCallback, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate

# Project Imports
from dataset import Dataset, CustomDataCollator


# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(0)



# Build models
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")



# Function: compute_metrics_ext
def compute_metrics_ext(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # decode from token_ids to strs
        preds_str = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels[labels == -100] = tokenizer.pad_token_id
        labels_str = tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        
        bert_score = bertscore.compute(predictions=preds_str, references=labels_str, model_type='microsoft/deberta-xlarge-mnli')
        rouge1_score_f1 = rouge.compute(predictions=preds_str, references=labels_str, use_aggregator=False, use_stemmer=False)

        return {'bert_score': sum(bert_score['f1']) / len(bert_score['f1']), 'rouge_score': sum(rouge1_score_f1['rouge1']) / len(rouge1_score_f1['rouge1'])}

    return compute_metrics



# Function: Create folder(s)
def _create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = os.path.join(folder, timestamp)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return timestamp, results_path



# Function: Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad), sum(p.numel() for p in model.parameters()
                                        if not p.requires_grad)



# Run the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to run the script.')

    # Processing parameters
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='Which gpus to use in CUDA_VISIBLE_DEVICES.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for dataloader.'
    )
    parser.add_argument(
        '--fp16',
        dest='fp16',
        action='store_true',
        help='Use 16-bit floating-point precision.',
    )
    parser.add_argument(
        '--no-fp16',
        dest='fp16',
        action='store_false',
        help='Use 16-bit floating-point precision.',
    )
    parser.set_defaults(fp16=False)

    # Directories and paths
    parser.add_argument(
        '--logdir',
        type=str,
        default='results',
        help='Directory where logs and models are to be stored.',
    )
    parser.add_argument(
        '--basedir',
        type=str,
        required=True,
        help='Directory where data is stored.',
    )

    # Model
    parser.add_argument(
        '--custom_tokenizer',
        type=str,
        default=None,
        help='Use custom tokenizer trained on the dataset. If the tokenizer from the model is to be used leave None.',
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='facebook/deit-tiny-distilled-patch16-224',
        choices=[
            'microsoft/beit-base-patch16-224-pt22k-ft22k',
            'google/vit-base-patch16-224', 'facebook/deit-tiny-patch16-224', 'facebook/deit-tiny-distilled-patch16-224'
        ],
        help='Encoder model to load.',
    )
    parser.add_argument(
        '--decoder',
        type=str,
        default='distilgpt2',
        choices=[
            'bert-base-uncased',
            'gpt2',
            'distilgpt2',
        ],
        help='Decoder model to load.',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Max length of inputs.',
    )

    # Training
    parser.add_argument(
        '--trainval',
        action='store_true',
        help='Train model on trainval split (for final submissions).'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='Load model from this checkpoint.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint.'
    )
    parser.add_argument(
        '--load_pretrained',
        action='store_true',
        help='Load pre trained model for fine tuning.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs.'
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        help='Batch size.',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Learning rate.',
    )
    parser.add_argument(
        '--decay',
        type=float,
        default=1e-6,
        help='Learning rate.',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience',
    )


    # Get arguments
    args = parser.parse_args()

    # Select pretrained model
    if args.resume or args.load_pretrained:
        assert args.resume != args.load_pretrained, 'Options resume and load_pretrained are mutually exclusive. Please choose only one.'
    
    # Get checkpoint
    if args.ckpt:
        assert (
            args.resume or args.load_pretrained
        ), 'When resuming training or loading pretrained model, you need to provide a checkpoint. When a checkpoint is provided, you need to select either the resume or the load_pretrained options.'


    # Select device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    # Create a folder to save logs
    timestamp, path = _create_folder(args.logdir)

    # Save experience arguments
    with open(os.path.join(path, 'train_params.txt'), 'w') as f:
        f.write(str(args))

    # WandB config
    os.environ['WANDB_PROJECT'] = 'imageclef23_cpt'
    os.environ['WANDB_DIR'] = path


    # Build model
    if args.ckpt:
        tokenizer_path = args.ckpt 
    else:
        if args.custom_tokenizer:
            tokenizer_path = args.custom_tokenizer
        else:
            tokenizer_path = args.decoder
    feature_extractor_path = args.ckpt if args.ckpt else args.encoder

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))

    feature_extractor = AutoImageProcessor.from_pretrained(feature_extractor_path)
    size = feature_extractor.size
    feature_extractor.save_pretrained(os.path.join(path, 'feature_extractor'))


    # Data
    # Train
    if args.trainval:
        tr_dtset = Dataset(
            os.path.join(args.basedir),
            os.path.join(args.basedir, 'ImageCLEFmedical_Caption_2023_caption_prediction_trainval_labels.csv'),
            img_processor=feature_extractor,
        )
    else:
        tr_dtset = Dataset(
            os.path.join(args.basedir, 'train'),
            os.path.join(args.basedir, 'ImageCLEFmedical_Caption_2023_caption_prediction_train_labels.csv'),
            img_processor=feature_extractor,
        )
    
    # Validation
    val_dtset = Dataset(
        os.path.join(args.basedir, 'valid'),
        os.path.join(args.basedir, 'ImageCLEFmedical_Caption_2023_caption_prediction_valid_labels.csv'),
        img_processor=feature_extractor,
    )


    # Model
    if args.ckpt:
        model = VisionEncoderDecoderModel.from_pretrained(args.ckpt)
        model.encoder.pooler = None 
        print('Model weights loaded from ' + args.ckpt)
    else:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            args.encoder,
            args.decoder,
            encoder_add_pooling_layer=False,
        )
        model.decoder.resize_token_embeddings(len(tokenizer))
        del model.config.encoder.label2id
        del model.config.encoder.id2label

        # Set special tokens
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        # Parameters for generation
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.max_length = args.max_length
        model.config.early_stopping = True

    model.train()

    model.config.to_json_file(os.path.join(path, 'conf.json'))

    trainable_params, non_trainable_params = count_parameters(model.encoder)
    print('Encoder params\n\t Trainable: %d \n\t Non trainable: %d\n' % (trainable_params, non_trainable_params))

    trainable_params, non_trainable_params = count_parameters(model.decoder)
    print('Decoder params\n\t Trainable: %d \n\t Non trainable: %d\n' % (trainable_params, non_trainable_params))

    # Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=path,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        weight_decay=args.decay,
        num_train_epochs=args.epochs,
        logging_dir=os.path.join(path, 'runs'),
        logging_strategy='epoch',
        save_strategy='epoch',
        predict_with_generate=True,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        log_level='debug',
        report_to='wandb',
        run_name=timestamp,
        metric_for_best_model='bert_score',
        greater_is_better=True
    )

    collator = CustomDataCollator(tokenizer, args.max_length)

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=tr_dtset,
        eval_dataset=val_dtset,
        callbacks=[EarlyStoppingCallback(args.patience)],
        compute_metrics=compute_metrics_ext(tokenizer)
    )
    trainer.train(resume_from_checkpoint=args.ckpt if args.resume else False)

    # Save best model
    print('Saving best model...')
    best_model_dir = trainer.state.best_model_checkpoint
    shutil.copytree(best_model_dir, os.path.join(path, 'checkpoint-best'))
    feature_extractor.save_pretrained(os.path.join(path, 'checkpoint-best'))
