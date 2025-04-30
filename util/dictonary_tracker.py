import os
import json
import pandas            as pd
import matplotlib.pyplot as plt

class GenericDictTracker:
    def __init__(self):
        self.items : list[dict[str, float]] = []

    def append(self, item: dict[str, float]) -> None:
        self.items.append(item)

    def last(self) -> dict[str, float]:
        return self.items[-1]

    def calculate_averages(self) -> dict[str, float]:
        averages = {}
        for item in self.items:
            for key, value in item.items():
                if key in averages:
                    averages[key].append(value)
                else:
                    averages[key] = [value]
        
        for key in averages:
            averages[key] = float(sum(averages[key]) / len(averages[key]))
        
        return averages

    def to_csv(self, fpath : str) -> None:
        df = pd.DataFrame(self.items)
        df.to_csv(fpath, index = False)
    
    def to_json(self, fpath : str) -> None:
        with open(fpath, 'w+') as fout:
            json.dump(self.items, fout, indent = 2)
    
    def plot(self, fpath : str, title = "") -> None:
        
        # plot the data
        for key_name in self.items[0].keys():
            vals = [x[key_name] for x in self.items]
            plt.plot(vals, label = key_name)
        
        plt.xlabel("Epoch")
        plt.ylabel("Values")
        plt.title(title)
        plt.legend()
        plt.savefig(fpath)
        plt.clf()


class AttachedTracker:
    def __init__(self, opt):
        self.root_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tracking')
        os.makedirs(self.root_dir, exist_ok = True)

        self.train_values = GenericDictTracker()
        self.eval_values  = GenericDictTracker()

        self.current_epoch   = 0
        self.metric_to_track = "structural_similarity_index_measure"

        self.is_current_best  = False
        self.current_best_val = 0 

    def append(self, train : GenericDictTracker, eval : GenericDictTracker) -> None:
        
        self.is_current_best = False

        self.train_values.append(
            train.calculate_averages()
        )
        self.eval_values.append(
            eval.calculate_averages()
        )

        latest_eval = self.eval_values.last()
        value_metric = latest_eval[self.metric_to_track]
        if value_metric > self.current_best_val:
            self.is_current_best  = True
            self.current_best_val = value_metric
            print(f"# Epoch {len(self.eval_values.items)} is Best {self.current_best_val}")
        
    def write(self) -> None:
        self.train_values.to_csv(
            os.path.join(self.root_dir, 'train_values.csv')
        )
        self.train_values.plot(
            os.path.join(self.root_dir, 'train.png'),
            title = 'Training Values'
        )

        self.eval_values.to_csv(
            os.path.join(self.root_dir, 'evaluation_values.csv')
        )
        self.eval_values.plot(
            os.path.join(self.root_dir, 'evaluation.png'),
            title = 'Evaluation Values'
        )

        



    

    