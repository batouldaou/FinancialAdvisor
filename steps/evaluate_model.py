import pickle
import os 


def save_model(model, config):
    save_path = os.path.join(config["output_dir"], f'{config["output_filename"]}.pickle')
    with open (save_path, 'wb') as file:
        pickle.dump(model,file)
    