from sklearn.model_selection import train_test_split
import json

from utilities import utilities
utils = utilities()


def create_dataset():
    # Load the training data
    data = utils.load_json_data("../../datasets/hover_train_release_v1.1.json")
    
    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save the split datasets to separate JSON files (optional)
    with open('../../datasets/roberta_custom/train_data.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open('../../datasets/roberta_custom/val_data.json', 'w') as f:
        json.dump(val_data, f, indent=4)
    
    
def main():
    create_dataset()
    
if __name__ == "__main__":
    main()