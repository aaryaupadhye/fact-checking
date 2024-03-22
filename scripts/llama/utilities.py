import sqlite3
import json
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd


class utilities:
    
    # Method to connect to HoVer database
    def connect_to_database(self, database_path):
        # Connect to the SQLite database.
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        return conn, cursor


    # Method to load JSON data
    def load_json_data(self, json_file_path):
        # Load the JSON data for HoVER.
        with open(json_file_path, "r") as json_file:
            json_data = json.load(json_file)
        return json_data
    
    
    # Method to load the 4-bit quantized model
    def load_model(self, model_name):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)   
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16,\
        quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer
    
        
    # Method to prepare input from HoVer
    def extract_evidence(self, nlp, cursor, supporting_facts):
        concat_evidence = ""
        evidence = ""
        extracted_evidence = []
        
        for fact in supporting_facts:
            id = fact[0]
            sentence_number = fact[1]
            cursor.execute("SELECT text FROM documents WHERE id=?", (id,))
            result = cursor.fetchone()
            if result:
                text = result[0]
                doc = nlp(text)
                sentences = [sentence.text for sentence in doc.sentences]
                if 0 <= sentence_number < len(sentences):
                    extracted_sentence = sentences[sentence_number]
                    extracted_evidence.append(extracted_sentence)

        for idx, sentence in enumerate(extracted_evidence):
            evidence += sentence+" "
            
        # Prepare evidence (title + evidence)
        concat_evidence = id+" : "+evidence
        
        return concat_evidence
    
    
    # Method for performance evaluation
    def eval(self, candidate_labels, true_labels, predicted_labels):
        cm = confusion_matrix(true_labels, predicted_labels, labels=candidate_labels)
        confusion_df = pd.DataFrame(cm, index=candidate_labels, columns=candidate_labels)
        
        true_labels_filtered = []
        predicted_labels_filtered = []

        for true_label, predicted_label in zip(true_labels, predicted_labels):
                if predicted_label != '':
                    true_labels_filtered.append(true_label)
                    predicted_labels_filtered.append(predicted_label)
                
        f1_sup = f1_score(true_labels_filtered, predicted_labels_filtered, pos_label="SUPPORTED")
        f1_not_sup= f1_score(true_labels_filtered, predicted_labels_filtered, pos_label="NOT_SUPPORTED")
        
        macro_f1 = (f1_sup + f1_not_sup) / 2
        return cm, confusion_df, accuracy_score(true_labels, predicted_labels), f1_sup, f1_not_sup, macro_f1
    
    
    # Method for assigning label from majority voting
    def find_mode(self, lst):
        # Filter out empty strings
        non_empty_lst = [s for s in lst if s != '']

        if not non_empty_lst:
            # If the list becomes empty after filtering, return an empty string
            return ''
        
        # Check if both "yes" and "no" are present in the non-empty list
        if "SUPPORTED" in non_empty_lst and "NOT_SUPPORTED" in non_empty_lst and len(non_empty_lst)==2:
            return ''

        # Check if there is only one distinct non-empty string in the list
        if len(set(non_empty_lst)) == 1:
            return non_empty_lst[0]

        # Find the mode in the non-empty list
        mode = max(set(non_empty_lst), key=non_empty_lst.count)

        return mode
    
    
    # Method to create log file
    def print_and_log(self, message, output_file, placeholder=None):
        # Print messages to console and log into a file.
        formatted_message = message if placeholder is None else message.format(placeholder)
        sys.stdout = sys.__stdout__
        print(formatted_message)  # Print to the console
        sys.stdout = output_file
        print(formatted_message, file=output_file)  # Write to the text file