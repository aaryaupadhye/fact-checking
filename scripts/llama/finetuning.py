import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, TrainingArguments
import os
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset
import matplotlib.pyplot as plt


# Method to load quantized model
def load_model(model_name, device):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False) 
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,\
    quantization_config=quantization_config, device_map={'':device})
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


# Method to print trainable parameters
def print_trainable_parameters(model):
    
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


    
def main():
    # Perform on available GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)


    dataset = load_dataset("json", data_files="../../datasets/llama_custom/finetuning_data.json")
    dataset = dataset["train"]
    print("created train dataset!")
    print(dataset)
     

    # Load the pre-trained model and tokenizer    
    base_model, tokenizer = load_model(model_name="../../datasets/Llama-2-13b-chat-hf", device=device)
 

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("loaded lora configs!")

    # Training Params
    train_params = TrainingArguments(
        output_dir="../../finetuned_models/llama/",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
        
    )
    
    print("loaded training params!")

    # Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params,
        max_seq_length=1024
    )
    
    print("initiated trainer!")
    
    
    # Print trainable parameters
    print_trainable_parameters(fine_tuning.model)

   
    # Train
    fine_tuning.train()
    print("done finetuning!")
    
    
    # Save Model
    fine_tuning.model.save_pretrained("llama")   
    print("saved finetuned model!")
       
    
    # Get training loss
    train_loss = []

    for log_history in fine_tuning.state.log_history:
        if 'loss' in log_history.keys():
            # Deal with trianing loss.
            train_loss.append(log_history['loss'])
            
    
    # Plot training loss
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Directory to save the plot image
    save_dir = "output"
    
    # Save the plot image
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    


if __name__ == "__main__":
    main()