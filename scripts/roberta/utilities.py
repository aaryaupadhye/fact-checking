import sqlite3
import json
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

class utilities:
    
    # Method to connect to HoVer database
    def connect_to_database(self, database_path):
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        return conn, cursor


    # Method to load JSON data
    def load_json_data(self, file_path):
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    
    
    # Method to prepare input from HoVer
    def prepare_input(self, nlp, cursor, supporting_facts, claim):
        concat_evidence = ""
        input_to_model = ""
        extracted_evidence = []
        
        for fact in supporting_facts:
            id = fact[0]
            sentence_number = fact[1]
            cursor.execute("SELECT text FROM documents WHERE id=?", (id,))
            result = cursor.fetchone()
            
            if result:
                # Get text from the result list
                text = result[0]
                
                # Process the text using Stanza to tokenize into sentences
                doc = nlp(text)
                sentences = [sentence.text for sentence in doc.sentences]
                
                # Check if the sentence number is within the valid range
                if 0 <= sentence_number < len(sentences):
                    extracted_sentence = sentences[sentence_number]
                    extracted_evidence.append(extracted_sentence)

        for idx, sentence in enumerate(extracted_evidence):
            concat_evidence += sentence+" "
            
        # Concatenate input (title + evidence + claim)
        input_to_model = id+" "+concat_evidence+" "+claim
        
        return input_to_model, concat_evidence
    
    
    # Method to load pre-trained model
    def load_model(self, model_name):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map = 'auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer


    # Method to tokenize and feed input to model
    def tokenize_input(self, tokenizer, input_text, model):
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        return inputs.to(model.device)
    

    # Method for evaluation
    def eval(self, candidate_labels, true_labels, predicted_labels):
        cm = confusion_matrix(true_labels, predicted_labels, labels=candidate_labels)
        confusion_df = pd.DataFrame(cm, index=candidate_labels, columns=candidate_labels)
        f1_sup = f1_score(true_labels, predicted_labels, pos_label="SUPPORTED")
        f1_not_sup= f1_score(true_labels, predicted_labels, pos_label="NOT_SUPPORTED")
        macro_f1 = (f1_sup + f1_not_sup) / 2
        return cm, confusion_df, accuracy_score(true_labels, predicted_labels), f1_sup, f1_not_sup, macro_f1


    # Method for hops analysis (sort predictions wrt hops)
    def analyze_hops(self, json_data, predicted_labels):   
        correct_predictions = [0, 0, 0]
        wrong_predictions = [0, 0, 0]
        
        for element, predicted_label in zip(json_data, predicted_labels):
            if element["num_hops"] == 2:
                if predicted_label == element["label"]:
                    correct_predictions[0] += 1
                else:
                    wrong_predictions[0] += 1
                    
            elif element["num_hops"] == 3:
                if predicted_label == element["label"]:
                    correct_predictions[1] += 1
                else:
                    wrong_predictions[1] += 1
                    
            elif element["num_hops"] == 4:
                if predicted_label == element["label"]:
                    correct_predictions[2] += 1
                else:
                    wrong_predictions[2] += 1
                
        return correct_predictions, wrong_predictions


    # Method to plot predictions vs hops
    def plot_hops_analysis(self, correct_predictions, wrong_predictions):
        hops = [2, 3, 4]
        total_predictions = [correct + wrong for correct, wrong in zip(correct_predictions, wrong_predictions)]
        correct_percentages = [(correct / total) * 100 for correct, total in zip(correct_predictions, total_predictions)]
        wrong_percentages = [(wrong / total) * 100 for wrong, total in zip(wrong_predictions, total_predictions)]

        bar_width = 0.35
        x = range(len(hops))

        fig, ax = plt.subplots()

        bar1 = ax.bar(x, correct_percentages, bar_width, label='Correct Predictions', color='cornflowerblue')
        bar2 = ax.bar(x, wrong_percentages, bar_width, bottom=correct_percentages, label='Wrong Predictions', color='lightcoral')

        ax.set_xlabel('Number of Hops')
        ax.set_ylabel('Percentage of Predictions')
        ax.set_title('Correct vs. Wrong Predictions')
        ax.set_xticks(x)
        ax.set_xticklabels(hops)
        ax.legend()

        # plot percentage of correct and wrong predictions wrt hops
        for i, (correct_percentage, wrong_percentage) in enumerate(zip(correct_percentages, wrong_percentages)):
            ax.text(i, correct_percentage / 2, f'{correct_percentage:.1f}%', ha='center', va='center', color='white')
            ax.text(i, correct_percentage + wrong_percentage / 2, f'{wrong_percentage:.1f}%', ha='center', va='center', color='white')

        plt.tight_layout()
        
        return fig
        
        
    # Method to create log file
    def print_and_log(self, message, output_file, placeholder=None):
        # Print messages to console and log into a file.
        formatted_message = message if placeholder is None else message.format(placeholder)
        sys.stdout = sys.__stdout__
        print(formatted_message)  # Print to the console
        sys.stdout = output_file
        print(formatted_message, file=output_file)  # Write to the text file