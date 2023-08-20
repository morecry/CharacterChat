from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from util import get_dataset, process_dataset

batch_size = 32

model_path = 'model/models/bert-base-uncased'
dataset_path = 'model/dataset/model_info_selecter_dataset.json'
output_dir = 'model/output/model_info_selecter'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)

dataset = get_dataset(dataset_path)
dataset = process_dataset(dataset, tokenizer)

train_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    save_strategy='steps',
    save_steps=200,
    learning_rate=2e-5,
    num_train_epochs=2.0,
    lr_scheduler_type='cosine',
    logging_steps=10,
    weight_decay=0.01
)

trainer = Trainer(
    model,
    train_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_state()
trainer.save_model()