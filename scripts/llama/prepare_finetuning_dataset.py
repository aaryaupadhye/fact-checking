import json
import stanza
import unicodedata

from utilities import utilities
utils = utilities()


# Method to prepare input from HoVer
def prepare_input(nlp, cursor, supporting_facts, claim):
    concat_evidence = ""
    input = ""
    extracted_evidence = []
    
    for fact in supporting_facts:
        id = fact[0]  # Get ID from the JSON supporting fact
        sentence_number = fact[1]  # Get sentence number from the JSON supporting fact
        extracted_evidence.append(id)   # Append title of article

        # Query the database to retrieve the text for the given ID
        cursor.execute("SELECT text FROM documents WHERE id=?", (unicodedata.normalize('NFD',id),))
        result = cursor.fetchone()

        if result:
            # Get text from the result list
            text = result[0]

            # Process the text using Stanza to tokenize into sentences
            doc = nlp(text)
            sentences = [sentence.text for sentence in doc.sentences]

            # Check if the sentence number is within the valid range
            if 0 <= sentence_number < len(sentences):
                # Extract the specific sentence
                extracted_sentence = sentences[sentence_number]
                extracted_evidence.append(extracted_sentence)

    # Prepare input to the model (title + evidence)
    concat_evidence = ""
    
    for idx, sentence in enumerate(extracted_evidence):
        concat_evidence += " : "+sentence+" "
        
    return concat_evidence



def main():
    # Connect to the HoVER database
    conn, cursor = utils.connect_to_database('../../datasets/wiki_wo_links.db')

    # Load the training data
    train_data = utils.load_json_data("../../datasets/hover_train_release_v1.1.json")

    # Initialize Stanza pipeline for sentence extraction
    nlp = stanza.Pipeline("en", processors="tokenize")

    input_list = []

    # Prepare dataset
    for element in train_data:
        claim = element["claim"]
        label = element["label"]
        supporting_facts = element["supporting_facts"]
        concat_evidence = prepare_input(nlp, cursor, supporting_facts, claim)  
        
        # Append generated prompts to the list for finetuning
        input_list.append({"text":"Given premise: \n"+concat_evidence+"\nClaim:\n"+claim+"\nTherefore the answer is: "+label})


    # Write the data to JSON
    with open('../../datasets/llama_custom/finetuning_data.json', 'w') as json_file:
        json.dump(input_list, json_file, indent=4)
        
        
    # Close the database connection
    conn.close()


if __name__ == "__main__":
    main()