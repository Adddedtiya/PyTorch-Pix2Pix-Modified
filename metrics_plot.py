import os
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_root = os.path.abspath('./checkpoints')

if __name__ == "__main__":
    print("### Pix7Mask Plot Mergers")

    parser = argparse.ArgumentParser(description="A list of checkpoint plots")

    parser.add_argument("--checkpoints", nargs='+', required = True)
    parser.add_argument("--mode", choices=['evaluation', 'train'], required = True)
    parser.add_argument("--variable", type = str, required = True)
    parser.add_argument("--out_plot", type = str, required = True)

    args = parser.parse_args()

    variable_name : str       = args.variable
    chk_names_lst : list[str] = args.checkpoints
    data_mode : str           = args.mode
    out_plot  : str           = args.out_plot

    combined_files_path : list[tuple[str, str]] = []
    for chname in chk_names_lst:
        dpath = os.path.join(base_root, chname, "tracking")
        
        if os.path.exists(dpath) == False:
            print(f"> '{chname}' does not exists")
            continue
            
        fpath = os.path.join(dpath, f"{data_mode}_values.csv")
        combined_files_path.append((chname, fpath))
    

    data_pairs : list[tuple[str, np.ndarray]] = []
    for (test_name, fpath) in combined_files_path:

        df = pd.read_csv(fpath)
        values = df[[variable_name]].values

        data_pairs.append((test_name, values))
    
    # plot the data
    for (test_name, values) in data_pairs:
        plt.plot(values, label = test_name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Values")
    plt.title(variable_name)
    plt.legend()
    plt.savefig(out_plot)
    

