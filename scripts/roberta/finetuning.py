import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import stanza
import matplotlib.pyplot as plt
import os

from utilities import utilities
utils = utilities()


def load_model(model_name, device):
    
    # Load model for finetuning with 2 labels
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2, ignore_mismatched_sizes=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

    
def encode_labels(true_labels):
    
    # Label encodings
    label_map = {"SUPPORTED": 1, "NOT_SUPPORTED": 0}
    encoded_labels = [label_map[label] for label in true_labels]
    return encoded_labels


# Class to create torch dataset for finetuning
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_model(model, train_loader, optimizer, device, num_epochs):
    
    # Train RoBERTa model
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            
def create_encodings(tokenizer, nlp, cursor, data):
    
    # Prepare encodings for finetuning
    input_list = []
    for element in data:
        claim = element["claim"]
        supporting_facts = element["supporting_facts"]
        input, evidence = utils.prepare_input(nlp, cursor, supporting_facts, claim)
        input_list.append(input)
        
    encodings = tokenizer(input_list, truncation=True, padding=True, max_length=512)
    return encodings
        
    
def main():
    # Perform on available GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Connect to the HoVER database
    conn, cursor = utils.connect_to_database('../../datasets/wiki_wo_links.db')  
    
    # Initialize Stanza pipeline for sentence extraction
    nlp = stanza.Pipeline("en", processors="tokenize")  
    
    # Load the pre-trained model and tokenizer    
    model, tokenizer = load_model("FacebookAI/roberta-large-mnli", device)
    
    # Load the training data
    train_data = utils.load_json_data("../../datasets/roberta_custom/train_data.json")    
    true_labels = [element["label"] for element in train_data]

    # Load the validation data
    val_data = utils.load_json_data("../../datasets/roberta_custom/val_data.json")
    val_true_labels = [element["label"] for element in val_data]    

    # Prepare encodings for finetuning
    encodings = create_encodings(tokenizer, nlp, cursor, train_data)
    val_encodings = create_encodings(tokenizer, nlp, cursor, val_data)
    
    encoded_labels = encode_labels(true_labels)
    val_encoded_labels = encode_labels(val_true_labels)

    print("done encodings!")

    # Create torch dataset
    train_dataset = ReviewDataset(encodings, encoded_labels)
    val_dataset = ReviewDataset(val_encodings, val_encoded_labels)
    
    print("created dataset!")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir='../../finetuned_models/roberta/',
        num_train_epochs=3,
        optim="paged_adamw_32bit",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='../../finetuned_models/roberta/logs',
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=1e-5,
        weight_decay=0.001,
        warmup_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="tensorboard"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Start training
    trainer.train()
    print("finetuning completed!")            
            
        
    # Save the finetuned model
    model.save_pretrained('../../finetuned_models/roberta/')
    tokenizer.save_pretrained('../../finetuned_models/roberta/')
    print("saved finetuned model!")
    
    # Check number of labels = 2
    print(model.config.num_labels)

    
    # Extract train loss    
    train_loss = []
    for log_history in trainer.state.log_history:
        if 'loss' in log_history.keys():
            train_loss.append(log_history['loss'])
    
    
    # Plot train loss
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Directory to save the plot image
    save_dir = "output"
    
    # Save the plot image
    plt.savefig(os.path.join(save_dir, "loss_plot.png")) 

    conn.close()


if __name__ == "__main__":
    main()