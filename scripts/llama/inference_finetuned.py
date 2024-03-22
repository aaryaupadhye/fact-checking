import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import stanza
from nltk.tokenize import sent_tokenize, word_tokenize
import time
import sys

from utilities import utilities
utils = utilities()


# Method to classify the result as SUPPORTED/NOT_SUPPORTED
def extract_label(text):
    
    sentences = sent_tokenize(text)
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        if 'NOT_SUPPORTED' in words or 'NOT_SUPPORTED"' in words or 'not_supported"' in words or 'not_supported' in words:
            return "NOT_SUPPORTED"
        elif 'SUPPORTED' in words or 'SUPPORTED"' in words or 'supported"' in words or 'supported' in words:
            return "SUPPORTED"
        

def main():
    
    # Perform on available GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'    
    
    # Connect to the HoVer database
    conn, cursor = utils.connect_to_database('../../datasets/wiki_wo_links.db')
    
    # Load the JSON data for HoVer
    json_data = utils.load_json_data("../../datasets/llama_custom/hover_sample.json")
    true_labels = [element["label"] for element in json_data]
    
    # Save the output logs
    output_file_path = "output/output_finetuned.txt"
    
    output_directory = os.path.dirname(output_file_path)

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
        
    # Initialize Stanza pipeline for sentence extraction
    nlp = stanza.Pipeline("en", processors="tokenize")
    
    # Load the finetuned model
    start_time = time.time()
    model, tokenizer = utils.load_model(model_name="../../finetuned_models/llama/")
    
    predicted_labels = []
    count = 1
    votes = []
    
    # Inference
    for element in json_data:        
        claim = element["claim"]
        supporting_facts = element["supporting_facts"]
        concat_evidence = utils.extract_evidence(nlp, cursor, supporting_facts)

            
        prompt1 = """   
        
        
        Given premise: 
        \n
        Misty Morning, Albert Bridge: It was composed by banjo player Jem Finer and featured on the band's fourth album, "Peace and Love". 
        Jem Finer: He was one of the founding members of The Pogues. 
        Longplayer: Longplayer is a self-extending composition by Jem Finer which is designed to continue for one thousand years. 
        James McNally (musician): He was previously a member of The Pogues and Storm (with Tom McManamon). 
        \n
        Claim:
        \n
        The composer of Longplayer was a banjo player in a band. The musician James McNally was in this band.
        \n
        Therefore the answer is: SUPPORTED
        
        
        Given premise: 
        \n
        Cardillac: Cardillac is an opera by Paul Hindemith in three acts and four scenes. 
        La donna del lago: La donna del lago (The Lady of the Lake) is an opera composed by Gioachino Rossini with a libretto by Andrea Leone Tottola (whose verses are described as "limpid" by one critic) based on the French translation of "The Lady of the Lake", a narrative poem written in 1810 by Sir Walter Scott, whose work continued to popularize the image of the romantic highlands. 
        Maria de Francesca-Cavazza: She can be heard and seen in the role of the Cardillac's daughter in Hindemith's opera Cardillac, conducted by Wolfgang Sawallisch, on a 1985 Munich DVD issued by Deutsche Grammophon. 
        Francesco Tortoli: He was the creator of sets for numerous productions including those for the world premieres of Rossini's "La gazzetta", "Otello", "Armida", "Mosè in Egitto", and "La donna del lago". 
        \n
        Claim:
        \n
        This work and a production Francesco Tortoli made sets for in 1987 are both operas. Maria de Francesca-Cavazza performed in this opera.
        \n
        Therefore the answer is: NOT_SUPPORTED
        

        Given premise: 
        \n
        Red, White &amp; Cr\u00fce: To coincide with the album's release, the band reunited with drummer Tommy Lee, who left the band in 1999. 
        Mike Tyson: Tyson won his first 19 professional fights by knockout, 12 of them in the first round. 
        Bobby Stewart: Bobby Stewart won the National Golden Gloves Tournament in 1974 as a light heavyweight, but he will be best remembered as the first trainer for Mike Tyson. 
        \n
        Claim: 
        \n
        Red, White & Crüe and this athlete both fight. The french fighter was trained by Bobby Stewart.
        \n
        Therefore the answer is: NOT_SUPPORTED
        
        
        Given premise: 
        \n
        Kristian Zahrtmann: Peder Henrik Kristian Zahrtmann, known as Kristian Zahrtmann, (31 March 1843 – 22 June 1917) was a Danish painter. 
        Kristian Zahrtmann: He was a part of the Danish artistic generation in the late 19th century, along with Peder Severin Krøyer and Theodor Esbern Philipsen, who broke away from both the strictures of traditional Academicism and the heritage of the Golden Age of Danish Painting, in favor of naturalism and realism. 
        Peder Severin: He is one of the best known and beloved, and the most colorful of the Skagen Painters, a community of Danish and Nordic artists who lived, gathered, or worked in Skagen, Denmark, especially during the final decades of the 19th century. 
        Ossian Elgstr: Elgström studied at the Royal Swedish Academy of Arts from 1906 to 1907, and then with Kristian Zahrtmann in 1907 and with Christian Krohg in 1908.
        \n
        Claim: 
        \n
        Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgström studied with in the early 1900s.
        \n
        Therefore the answer is: SUPPORTED
        
        
        Given premise: 
        \n
        DAF XF: The DAF XF is a range of trucks produced by the Dutch manufacturer DAF since 1997. 
        DAF XF: All right hand drive versions of the XF are assembled at Leyland Trucks in the UK.
        DAF Trucks: Some of the truck models sold with the DAF brand are designed and built by Leyland Trucks at their Farington plant in Leyland near Preston, England.
        \n
        Claim: 
        \n
        The DAF XF articulated truck is produced by Leyland Trucks at their Farington plant.
        \n
        Therefore the answer is: NOT_SUPPORTED
        
        
        Given premise: 
        \n
        WWE Super Tuesday: Super Tuesday was a 1-hour professional wrestling television special event, produced by the World Wrestling Entertainment (WWE) that took place on 12 November 2002 (which was taped November 4 & 5) at the Fleet Center in Boston, Massachusetts and Verizon Wireless Arena in Manchester, New Hampshire, which featured matches from both Raw and SmackDown. 
        TD Garden: TD Garden, often called Boston Garden and \"The Garden\", is a multi-purpose arena in Boston, Massachusetts.
        TD Garden: It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
        \n
        Claim: 
        \n
        WWE Super Tuesday took place at an arena that currently goes by the name TD Garden.
        \n
        Therefore the answer is: SUPPORTED
        
        
        Given premise:\n '{test_evidence}' \n
        Claim:\n {test_claim} \n
        
        """.format(test_evidence = concat_evidence, test_claim = claim)

        
        prompt2 = """   
        
        Given premise: 
        \n
        Cardillac: Cardillac is an opera by Paul Hindemith in three acts and four scenes. 
        La donna del lago: La donna del lago (The Lady of the Lake) is an opera composed by Gioachino Rossini with a libretto by Andrea Leone Tottola (whose verses are described as "limpid" by one critic) based on the French translation of "The Lady of the Lake", a narrative poem written in 1810 by Sir Walter Scott, whose work continued to popularize the image of the romantic highlands. 
        Maria de Francesca-Cavazza: She can be heard and seen in the role of the Cardillac's daughter in Hindemith's opera Cardillac, conducted by Wolfgang Sawallisch, on a 1985 Munich DVD issued by Deutsche Grammophon. 
        Francesco Tortoli: He was the creator of sets for numerous productions including those for the world premieres of Rossini's "La gazzetta", "Otello", "Armida", "Mosè in Egitto", and "La donna del lago". 
        \n
        Claim:
        \n
        This work and a production Francesco Tortoli made sets for in 1987 are both operas. Maria de Francesca-Cavazza performed in this opera.
        \n
        Therefore the answer is: NOT_SUPPORTED
        
        
        Given premise: 
        \n
        Red, White &amp; Cr\u00fce: To coincide with the album's release, the band reunited with drummer Tommy Lee, who left the band in 1999. 
        Mike Tyson: Tyson won his first 19 professional fights by knockout, 12 of them in the first round. 
        Bobby Stewart: Bobby Stewart won the National Golden Gloves Tournament in 1974 as a light heavyweight, but he will be best remembered as the first trainer for Mike Tyson. 
        \n
        Claim: 
        \n
        Red, White & Crüe and this athlete both fight. The french fighter was trained by Bobby Stewart.
        \n
        Therefore the answer is: NOT_SUPPORTED
        
        
        Given premise: 
        \n
        Misty Morning, Albert Bridge: It was composed by banjo player Jem Finer and featured on the band's fourth album, "Peace and Love". 
        Jem Finer: He was one of the founding members of The Pogues. 
        Longplayer: Longplayer is a self-extending composition by Jem Finer which is designed to continue for one thousand years. 
        James McNally (musician): He was previously a member of The Pogues and Storm (with Tom McManamon). 
        \n
        Claim:
        \n
        The composer of Longplayer was a banjo player in a band. The musician James McNally was in this band.
        \n
        Therefore the answer is: SUPPORTED   
        
        
        Given premise: 
        \n
        DAF XF: The DAF XF is a range of trucks produced by the Dutch manufacturer DAF since 1997. 
        DAF XF: All right hand drive versions of the XF are assembled at Leyland Trucks in the UK.
        DAF Trucks: Some of the truck models sold with the DAF brand are designed and built by Leyland Trucks at their Farington plant in Leyland near Preston, England.
        \n
        Claim: 
        \n
        The DAF XF articulated truck is produced by Leyland Trucks at their Farington plant.
        \n
        Therefore the answer is: NOT_SUPPORTED


        Given premise: 
        \n
        WWE Super Tuesday: Super Tuesday was a 1-hour professional wrestling television special event, produced by the World Wrestling Entertainment (WWE) that took place on 12 November 2002 (which was taped November 4 & 5) at the Fleet Center in Boston, Massachusetts and Verizon Wireless Arena in Manchester, New Hampshire, which featured matches from both Raw and SmackDown. 
        TD Garden: TD Garden, often called Boston Garden and \"The Garden\", is a multi-purpose arena in Boston, Massachusetts.
        TD Garden: It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
        \n
        Claim: 
        \n
        WWE Super Tuesday took place at an arena that currently goes by the name TD Garden.
        \n
        Therefore the answer is: SUPPORTED
        
        
        Given premise: 
        \n
        Kristian Zahrtmann: Peder Henrik Kristian Zahrtmann, known as Kristian Zahrtmann, (31 March 1843 – 22 June 1917) was a Danish painter. 
        Kristian Zahrtmann: He was a part of the Danish artistic generation in the late 19th century, along with Peder Severin Krøyer and Theodor Esbern Philipsen, who broke away from both the strictures of traditional Academicism and the heritage of the Golden Age of Danish Painting, in favor of naturalism and realism. 
        Peder Severin: He is one of the best known and beloved, and the most colorful of the Skagen Painters, a community of Danish and Nordic artists who lived, gathered, or worked in Skagen, Denmark, especially during the final decades of the 19th century. 
        Ossian Elgstr: Elgström studied at the Royal Swedish Academy of Arts from 1906 to 1907, and then with Kristian Zahrtmann in 1907 and with Christian Krohg in 1908.
        \n
        Claim: 
        \n
        Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgström studied with in the early 1900s.
        \n
        Therefore the answer is: SUPPORTED
        
        
        Given premise: '{test_evidence}'
        Claim: {test_claim}
        
        """.format(test_evidence = concat_evidence, test_claim = claim)
        
        
        prompt3 = """   
        
        Given premise: 
        \n
        Red, White &amp; Cr\u00fce: To coincide with the album's release, the band reunited with drummer Tommy Lee, who left the band in 1999. 
        Mike Tyson: Tyson won his first 19 professional fights by knockout, 12 of them in the first round. 
        Bobby Stewart: Bobby Stewart won the National Golden Gloves Tournament in 1974 as a light heavyweight, but he will be best remembered as the first trainer for Mike Tyson. 
        \n
        Claim: 
        \n
        Red, White & Crüe and this athlete both fight. The french fighter was trained by Bobby Stewart.
        \n
        Therefore the answer is: NOT_SUPPORTED
        
        Given premise: 
        \n
        Misty Morning, Albert Bridge: It was composed by banjo player Jem Finer and featured on the band's fourth album, "Peace and Love". 
        Jem Finer: He was one of the founding members of The Pogues. 
        Longplayer: Longplayer is a self-extending composition by Jem Finer which is designed to continue for one thousand years. 
        James McNally (musician): He was previously a member of The Pogues and Storm (with Tom McManamon). 
        \n
        Claim:
        \n
        The composer of Longplayer was a banjo player in a band. The musician James McNally was in this band.
        \n
        Therefore the answer is: SUPPORTED
        
        
        Given premise: 
        \n
        DAF XF: The DAF XF is a range of trucks produced by the Dutch manufacturer DAF since 1997. 
        DAF XF: All right hand drive versions of the XF are assembled at Leyland Trucks in the UK.
        DAF Trucks: Some of the truck models sold with the DAF brand are designed and built by Leyland Trucks at their Farington plant in Leyland near Preston, England.
        \n
        Claim: 
        \n
        The DAF XF articulated truck is produced by Leyland Trucks at their Farington plant.
        \n
        Therefore the answer is: NOT_SUPPORTED
        
        
        Given premise: 
        \n
        Kristian Zahrtmann: Peder Henrik Kristian Zahrtmann, known as Kristian Zahrtmann, (31 March 1843 – 22 June 1917) was a Danish painter. 
        Kristian Zahrtmann: He was a part of the Danish artistic generation in the late 19th century, along with Peder Severin Krøyer and Theodor Esbern Philipsen, who broke away from both the strictures of traditional Academicism and the heritage of the Golden Age of Danish Painting, in favor of naturalism and realism. 
        Peder Severin: He is one of the best known and beloved, and the most colorful of the Skagen Painters, a community of Danish and Nordic artists who lived, gathered, or worked in Skagen, Denmark, especially during the final decades of the 19th century. 
        Ossian Elgstr: Elgström studied at the Royal Swedish Academy of Arts from 1906 to 1907, and then with Kristian Zahrtmann in 1907 and with Christian Krohg in 1908.
        \n
        Claim: 
        \n
        Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgström studied with in the early 1900s.
        \n
        Therefore the answer is: SUPPORTED
        
        
        Given premise: 
        \n
        WWE Super Tuesday: Super Tuesday was a 1-hour professional wrestling television special event, produced by the World Wrestling Entertainment (WWE) that took place on 12 November 2002 (which was taped November 4 & 5) at the Fleet Center in Boston, Massachusetts and Verizon Wireless Arena in Manchester, New Hampshire, which featured matches from both Raw and SmackDown. 
        TD Garden: TD Garden, often called Boston Garden and \"The Garden\", is a multi-purpose arena in Boston, Massachusetts.
        TD Garden: It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
        \n
        Claim: 
        \n
        WWE Super Tuesday took place at an arena that currently goes by the name TD Garden.
        \n
        Therefore the answer is: SUPPORTED
        
            
        Given premise: 
        \n
        Cardillac: Cardillac is an opera by Paul Hindemith in three acts and four scenes. 
        La donna del lago: La donna del lago (The Lady of the Lake) is an opera composed by Gioachino Rossini with a libretto by Andrea Leone Tottola (whose verses are described as "limpid" by one critic) based on the French translation of "The Lady of the Lake", a narrative poem written in 1810 by Sir Walter Scott, whose work continued to popularize the image of the romantic highlands. 
        Maria de Francesca-Cavazza: She can be heard and seen in the role of the Cardillac's daughter in Hindemith's opera Cardillac, conducted by Wolfgang Sawallisch, on a 1985 Munich DVD issued by Deutsche Grammophon. 
        Francesco Tortoli: He was the creator of sets for numerous productions including those for the world premieres of Rossini's "La gazzetta", "Otello", "Armida", "Mosè in Egitto", and "La donna del lago". 
        \n
        Claim:
        \n
        This work and a production Francesco Tortoli made sets for in 1987 are both operas. Maria de Francesca-Cavazza performed in this opera.
        \n
        Therefore the answer is: NOT_SUPPORTED
        
        
        Given premise: '{test_evidence}'
        Claim: {test_claim}
        
        """.format(test_evidence = concat_evidence, test_claim = claim)
        

        prompts = [prompt1, prompt2, prompt3]
        
        
        for p in prompts:
            input_ids = tokenizer(p, return_tensors="pt").input_ids.to(model.device)

            # Generate text using the model
            output = model.generate(input_ids, 
                                    num_return_sequences=1, max_new_tokens=20)

            # Decode and print the generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Remove the prompt and period from the generated text
            generated_text = generated_text.replace(p, "").strip()
            generated_text = generated_text.replace(".", "")
        
            # Extract label from result
            predicted_label = extract_label(generated_text)
            
            # Log the results
            with open(output_file_path, "a") as output_file:
                
                utils.print_and_log(count,output_file=output_file)
                count = count+1
                utils.print_and_log("\nClaim: {}", output_file, claim)
                utils.print_and_log("\nEvidence: {}", output_file, concat_evidence)
                utils.print_and_log("\nResult:\n {}", output_file, generated_text)
                utils.print_and_log("\nPredicted Label: {}", output_file, predicted_label)
                utils.print_and_log("\nActual label: {}", output_file, element['label'])
                utils.print_and_log("\n--------------\n\n", output_file=output_file) 
            
            votes.append(predicted_label)
            # Replace None values with an empty string during the comparison
            votes = ['' if label is None else label for label in votes]
        
        predicted_labels.append(utils.find_mode(votes))
        
        # Log the results
        with open(output_file_path, "a") as output_file:
            utils.print_and_log("\nVotes: {}", output_file, votes)
            utils.print_and_log("\nLabel: {}", output_file, utils.find_mode(votes))
        
        votes.clear()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Log the results
    with open(output_file_path, "a") as output_file:
        utils.print_and_log("\n\nTRUE LABELS: \n{}", output_file, true_labels)
        utils.print_and_log("\n\nPREDICTED LABELS:\n{}", output_file, predicted_labels)
        utils.print_and_log("\nTime taken (seconds): {}", output_file, elapsed_time)
        
    
    # Evaluation
    candidate_labels = ['SUPPORTED', 'NOT_SUPPORTED']
    cm, confusion_df, accuracy, f1_sup, f1_not_sup, macro_f1 = utils.eval(candidate_labels, true_labels, predicted_labels)
    
     # Plot heatmap
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues', xticklabels=candidate_labels, yticklabels=candidate_labels)
    plt.savefig("output/heatmap_finetuned.png")
        
    # Log the results
    with open(output_file_path, "a") as output_file:
        utils.print_and_log("\nLLAMA Confusion Matrix:\n", output_file)
        utils.print_and_log(confusion_df, output_file)
        utils.print_and_log("\nAccuracy: {}", output_file, accuracy)
        utils.print_and_log("\nF-1 Score 'SUPPORTED': {}", output_file, f1_sup)
        utils.print_and_log("\nF-1 Score 'NOT_SUPPORTED': {}", output_file, f1_not_sup)
        utils.print_and_log("\nMacro F-1 Score: {}", output_file, macro_f1)
        utils.print_and_log("\nF1-Score: {}", output_file, (f1_sup+f1_not_sup)/2)

        utils.print_and_log("done!", output_file)

    sys.stdout = sys.__stdout__
    print("done last iteration!")
    
    # Close the database connection
    conn.close()


if __name__ == "__main__":
    main()