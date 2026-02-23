from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt


class DisplayResults:
    def __init__(self, input_dir:str ="output/graded", output_dir: str = "output/summaries", dir: str =""):
        self.output_dir = output_dir + "/" + dir
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.input_dir = input_dir + "/" + dir
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
                for framework, framework_data in data.items():
                    if f"{model} X {framework}" not in results:
                        results[f"{model} X {framework}"] = []
                    results[f"{model} X {framework}"].append(framework_data)
        return results

    def get_preformance(self) -> dict:
        """
        Docstring for get_preformance
        
        :return: the dry performance metrics for each model
        :rtype: dict
        """
        performance = {}
        for agent, data in self.results.items():
            performance[agent] = {}
            summary = [dat["grading_summary"] for dat in data]
            accuracy = [summ.get("accuracy", 0) for summ in summary]
            correct_answers = [sum.get("correct_answers", 0) for sum in summary]
            avg_time = [summ.get("summary", {}).get("avg_execution_time", 0) for summ in data]
            failed_runs = [summ.get("summary", {}).get("failed_runs", 0) for summ in data]
            performance[agent] = {
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
        for agent, agents_data in self.results.items():
            literary_details[agent] = []
            for data in agents_data:
                grading_summary = data.get("grading_summary", {})
                details = grading_summary.get("literary_details")
                literary_details[agent].append(details)
        return literary_details
    
    def save_plot_performance(self):
        """
        plots the performance metrics for each model
        
        :return: None
        """
        performance = self.get_preformance()

        for metric in ["accuracy", "correct_answers", "avg_execution_time","failed_runs"]:
            plt.figure(figsize=(10, 6))
            # Define unique colors for each model
            model_colors = {
                'gpt-oss-120b': '#1f77b4',  # blue
                'gpt-oss-20b': '#ff7f0e',   # orange
                'Meta-Llama-3': '#2ca02c',        # green
                'Mistral-Small-3.2-24B-Instruct-2506': '#d62728'           # red
            }

            agents = sorted(performance.keys())
            
            # Check if any metric is a list with length > 1
            has_multiple_values = any(len(performance[agent][metric]) > 1 for agent in agents)
            
            if has_multiple_values:
                # Use box chart for multiple values
                print(f"Plotting boxplot for metric: {metric}")
                data_to_plot = [performance[agent][metric] for agent in agents]
                plt.boxplot(data_to_plot, tick_labels=agents, positions=range(len(agents)))
            else:
                # Use bar chart for single values
                for agent in agents:
                    agent_data = performance[agent]
                    base_model = agent.split(' ')[0]
                    color = model_colors.get(base_model, '#7f7f7f')
                    plt.bar(agent, agent_data[metric], label=agent, color=color)

            plt.title(f'Model Performance: {metric.replace("_", " ").title()}')
            plt.xlabel('Model - Framework')
            plt.ylabel(metric.replace("_", " ").title())
            plt.xticks(rotation=45, ha='right')
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

    def plot_together(self, united_file: str):
        """
        plots all agets together in a single plot for comparison, whilst stacking 
        the performance levels for each model-framework combination
        
        :param united_file: the file to save the united plot
        :type united_file: str
        :return: None
        """
        data = {}
        with open(united_file, 'r') as f:
            data = json.load(f)
        # Prepare data for stacked bar chart
        models = sorted(list(data.keys()))
        frameworks = set()
        for model_data in data.values():
            frameworks.update(model_data.keys())
        frameworks = sorted(list(frameworks))

        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 7))
        width = 0.8
        colors = ["#32935a", "#a77729", "#982f24"]  # green, orange, red for levels 1, 2, 3

        # Create x positions for each model-framework combination
        combinations = []
        x_positions = []
        pos = 0
        
        for model in models:
            for framework in frameworks:
                combinations.append(f"{model}\n{framework}")
                x_positions.append(pos)
                pos += 1
            pos += 1.5  # Add gap between different models

        # Plot stacked bars for each combination
        for i, (model, framework) in enumerate([(m, f) for m in models for f in frameworks]):
            level_1 = data[model].get(framework, {}).get("1", {}).get("avg_correct", 0)
            level_2 = data[model].get(framework, {}).get("2", {}).get("avg_correct", 0)
            level_3 = data[model].get(framework, {}).get("3", {}).get("avg_correct", 0)

            ax.bar(x_positions[i], level_1, width, 
               label='Level 1' if i == 0 else '', color=colors[0], alpha=0.8)
            ax.bar(x_positions[i], level_2, width, bottom=level_1,
               label='Level 2' if i == 0 else '', color=colors[1], alpha=0.8)
            ax.bar(x_positions[i], level_3, width,
               bottom=level_1 + level_2,
               label='Level 3' if i == 0 else '', color=colors[2], alpha=0.8)

        ax.set_xlabel('Model - Framework', fontsize=12)
        ax.set_ylabel('Number of Correct Answers', fontsize=12)
        ax.set_title('Model Performance by Framework and Difficulty Level', fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(combinations, rotation=45, ha='right', fontsize=9)
        ax.legend(title='Levels', loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/stacked_performance.png', dpi=300)
        plt.close()
        print(f"Stacked performance plot saved to {self.output_dir}/stacked_performance.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <directory/file_path>")
        sys.exit(1)
    dir = sys.argv[1]
    display = DisplayResults(dir=dir)
    display.plot_together("output/connected_comparisons_20260219_121311.json")
    # display.save_plot_performance()
    display.save_description()
