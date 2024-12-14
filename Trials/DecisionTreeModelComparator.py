import pandas as pd

class ModelComparator:
    def __init__(self, results_map):
        self.results_map = results_map

    def compare(self):
        comparison_data = []

        for model_name, folder_path in self.results_map.items():
            metrics_csv_path = os.path.join(folder_path, "metrics.csv")
            if not os.path.exists(metrics_csv_path):
                raise FileNotFoundError(f"Metrics file not found in folder: {folder_path}")

            metrics_df = pd.read_csv(metrics_csv_path)
            metrics_row = {"Model": model_name}
            for _, row in metrics_df.iterrows():
                metrics_row[row["Metric"]] = row["Value"]

            comparison_data.append(metrics_row)

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df