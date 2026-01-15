import matplotlib.pyplot as plt
import json
from pathlib import Path


class DisplayResults:
    def __init__(self, input_dir:str ="output/graded", output_dir: str = "output/summaries"):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.results = self.get_results()

    def get_results(self) -> dict:
        """
        Docstring for get_results
        
        :param directory: the directory containing graded result files
        :type directory: str
        :return: the compiled results from all graded files
        :rtype: dict
        """
        results = {}
        for file in Path(self.input_dir).glob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                model = file.stem.split('_')[1]
                results[model] = data
        return results

    def get_preformance(self) -> dict:
        """
        Docstring for get_preformance
        
        :return: the dry performance metrics for each model
        :rtype: dict
        """
        performance = {}
        for model, data in self.results.items():
            performance[model] = {}
            for framework, framework_data in data.items():
                summary = framework_data.get("grading_summary", {})
                accuracy = summary.get("accuracy", 0)
                correct_answers = summary.get("correct_answers", 0)
                avg_time = framework_data.get("summary", {}).get("avg_execution_time", 0)
                failed_runs = framework_data.get("summary", {}).get("failed_runs", 0)
                performance[model][framework] = {
                    "accuracy": accuracy,
                    "correct_answers": correct_answers,
                    "avg_execution_time": avg_time,
                    "failed_runs": failed_runs
                }
        return performance
    
    def get_literary_details(self) -> dict:
        """
        Docstring for get_literary_details
        
        :return: the literary details for each model
        :rtype: dict
        """
        literary_details = {}
        for model, data in self.results.items():
            for framework, framework_data in data.items():
                summary = framework_data.get("grading_summary", {})
                details = summary.get("literary_details", {})
                literary_details[model+" X " + framework] = details
        return literary_details
    
    def save_plot_performance(self):
        """
        plots the performance metrics for each model
        
        :return: None
        """
        performance = self.get_preformance()
        
        for metric in ["accuracy", "correct_answers", "avg_execution_time","failed_runs"]:
            plt.figure(figsize=(10, 6))
            for model, frameworks in performance.items():
                values = [frameworks[fw][metric] for fw in frameworks]
                plt.bar([f"{model} - {fw}" for fw in frameworks], values, label=model)
            
            plt.title(f'Model Performance: {metric.replace("_", " ").title()}')
            plt.xlabel('Model - Framework')
            plt.ylabel(metric.replace("_", " ").title())
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{metric}_performance.png')
            plt.close()

    def save_description(self):
        """
        saves the literary details to a single JSON file
        
        :param output_file: the file to save literary details
        :type output_file: str
        :return: None
        """
        output_file = f"{self.output_dir}/litereary_details.json"
        with open(output_file, 'w') as f:
            json.dump(self.get_literary_details(), f, indent=2)
        print(f"Literary details saved to {output_file}")

if __name__ == "__main__":
    display = DisplayResults("output/graded/lvl3", "output/summaries/lvl3")
    display.save_plot_performance()
    display.save_description()
