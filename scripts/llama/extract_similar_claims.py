from nltk.tokenize import word_tokenize
import json


def prepare_search_space():
    # Prepare base database
    with open('../../datasets/hover_train_release_v1.1.json','r') as file:
        train = json.load(file)
        
    with open('../../datasets/hover_dev_release_v1.1.json','r') as file:
        dev = json.load(file)
        
    search_space = train+dev
    
    return search_space


def load_input_database():
    # Load input data
    with open('../../datasets/llama_custom/hover_prompt.json','r') as file:
        input_data = json.load(file)
        
        return input_data
    
    
# Method to extract similar 'SUPPORTED' claims in the search space for 'NOT_SUPPORTED' claims in the prompt
def search_similar_claims(search_space, input_data):
    found = []
    count = 0
    for idx, element in enumerate(input_data):
        
        if element['label'] == "NOT_SUPPORTED":
            
            claim = element['claim']
            claim_words = word_tokenize(claim)
            
            matches = []
            for entry in search_space:
                check_claim = word_tokenize(entry['claim'])        
                
                if len(list(set(claim_words).intersection(check_claim))) > 0.7 * len(list(set(claim_words))) and entry['label'] == "SUPPORTED":
                    matches.append(entry['claim'])
                    continue
            
            if matches:   
                found.append({'index': count, 'relative index':idx, 'claim':claim , 'similar_supported_claims':matches})
                count = count + 1                       

    # Save found claims in JSON
    with open('../../datasets/llama_custom/similar_claims.json','w') as file:
        json.dump(found,file,indent=2)
    
    
def main():
    
    # Prepare search space
    search_space = prepare_search_space()
    
    # Load input data 
    input_data = load_input_database()
    
    # Extract similar claims
    search_similar_claims(search_space, input_data)
    
    print("done!")
    

if __name__ == "__main__":
    main()