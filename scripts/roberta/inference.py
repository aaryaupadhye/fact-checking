import os
import torch
import torch.nn.functional as F
import numpy as np
import stanza
import seaborn as sns
import matplotlib.pyplot as plt

from utilities import utilities
utils = utilities()


def predict_class(model, inputs):
        with torch.no_grad():
            logits = model(**inputs).logits
            
        # Apply softmax to get a set of probabilities
        probabilities = F.softmax(logits, dim=1)
        
        for probs in probabilities:
            
            # Classify 'SUPPORTED' if 2 (entailment) highest, else 'NOT_SUPPORTED' (contradiction/ neutral)
            if (probs[2] > probs[1]) and (probs[2] > probs[0]):
                predicted_class = "SUPPORTED"
            else:
                predicted_class = "NOT_SUPPORTED"
                
        return predicted_class


def main():
    
    # Perform on available GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Connect to the HoVer database
    conn, cursor = utils.connect_to_database('../../datasets/wiki_wo_links.db')

    # Load the JSON data for HoVER
    json_data = utils.load_json_data("../../datasets/hover_dev_release_v1.1.json")
    true_labels = [element["label"] for element in json_data]
    
    # Save output
    output_file_path = "output/output.txt"
    
    output_directory = os.path.dirname(output_file_path)

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Initialize Stanza pipeline for sentence extraction
    nlp = stanza.Pipeline("en", processors="tokenize")
    
    # Load RoBERTa pre-trained model and tokenizer    
    model, tokenizer = utils.load_model(model_name="FacebookAI/roberta-large-mnli")
    
    # # For Inference on BART
    # model, tokenizer = utils.load_model(model_name="facebook/bart-large-mnli")
    
    predicted_labels = []
    count = 1
    
    # Inference
    for element in json_data:        
        claim = element["claim"]
        supporting_facts = element["supporting_facts"]
        input_to_model, evidence = utils.prepare_input(nlp, cursor, supporting_facts, claim)

        inputs = utils.tokenize_input(tokenizer, input_to_model, model)
        predicted_class = predict_class(model, inputs)
        predicted_labels.append(predicted_class)

        with open(output_file_path, "a") as output_file:
            utils.print_and_log(count,output_file=output_file)
            count=count+1
            utils.print_and_log("\nTrue label: {}", output_file, element["label"])
            utils.print_and_log("\nRoberta Predicted label: {}", output_file, predicted_class)
            utils.print_and_log("\nThe claim '{}'", output_file, claim)
            utils.print_and_log("is '{}'", output_file, predicted_class)
            utils.print_and_log("based on evidence - '{}'", output_file, evidence)
            utils.print_and_log("\n-----------------\n",output_file)
       
    # Evaluation
    candidate_labels = ['SUPPORTED', 'NOT_SUPPORTED']
    cm, confusion_df, acc, f1_sup, f1_not_sup, macro_f1 = utils.eval(candidate_labels, true_labels, predicted_labels)
    
    with open(output_file_path, "a") as output_file:
        utils.print_and_log("\nConfusion Matrix:\n {}", output_file, confusion_df)
        utils.print_and_log("\nAccuracy score: {}", output_file, acc)
        utils.print_and_log("\nF-1 Score SUPPORTED: {}", output_file, f1_sup)
        utils.print_and_log("\nF-1 Score NOT_SUPPORTED: {}", output_file, f1_not_sup)
        utils.print_and_log("\nMacro F-1: {}", output_file, macro_f1)
    
    # Plot heatmap
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues', xticklabels=candidate_labels, yticklabels=candidate_labels)
    plt.savefig("output/heatmap.png")
    
    # Hops Analysis
    correct_predictions, wrong_predictions = utils.analyze_hops(json_data, predicted_labels)
    
    with open(output_file_path, "a") as output_file:
        utils.print_and_log("\nCorrect predictions by Roberta: {}", output_file, correct_predictions)
        utils.print_and_log("\nWrong predictions by Roberta: {}", output_file, wrong_predictions)

    # Vizualization of hops analysis
    fig = utils.plot_hops_analysis(correct_predictions, wrong_predictions)  
    fig.savefig("output/hops.png")  

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    main()