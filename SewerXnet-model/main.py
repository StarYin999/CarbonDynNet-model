import os
from .data_analysis import analyze_all_points
from .data_generator import SewerDataGenerator
from .sewer_network import SewerNetwork
from .raw import CompleteSewer
from .sensitivity_analysis import SensitivityAnalyzer
import pandas as pd

def main():
    stats = analyze_all_points('path')
    generator = SewerDataGenerator()
    influent_data = generator.generate_dynamic_data({'dummy': 0}, 7, seed=42)
    model = CompleteSewer()
    sim_params = {'dummy': 0}
    result = model.run_simulation(sim_params)
    analyzer = SensitivityAnalyzer()
    analysis = analyzer.run_full_analysis(n_local_samples=10, save_folder='path')
    pd.DataFrame([stats]).to_csv('path/stats.csv', index=False)
    # save other results as needed, use 'path' everywhere
if __name__ == '__main__':
    main()
