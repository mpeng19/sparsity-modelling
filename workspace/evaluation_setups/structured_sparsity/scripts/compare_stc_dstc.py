import os
import sys
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from collections import defaultdict
import itertools
import re

try:
    from parse_timeloop_stats_file import parse_stats_txt
except ImportError:
    print("ERROR: Cannot find parse_timeloop_stats_file.py.")
    print("Ensure it is in the same directory as this script.")
    sys.exit(1)

TARGET_DENSITIES = {
    0.25: "WD-0.25",
    0.3333: "WD-0.3333",
    0.5: "WD-0.5",
    1.0: "WD-1.0"
}
DENSITY_SORT_ORDER = [0.25, 0.3333, 0.5, 1.0]

TARGET_SCHEMES = {
    "DSTC": "DSTC-RF2x-24-bandwidth",
    "STC": "STC-RF2x-24-bandwidth"
}
SCHEME_SORT_ORDER = ["STC", "DSTC"]

STATS_FILENAME = "timeloop-model.stats.txt"

scripts_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
base_output_dir = os.path.abspath(os.path.join(scripts_dir, '..', 'outputs'))

def main():
    parser = argparse.ArgumentParser(description=f"Compare STC vs DSTC results for a given workload across multiple densities.")
    parser.add_argument('--workload', type=str, required=True,
                        choices=['resnet50_selected', 'alexnet_selected', 'mobilenetv2_selected'],
                        help='Name of the model workload to compare (e.g., resnet50_selected).')
    args = parser.parse_args()
    workload = args.workload

    #Construct base paths using the parsed workload
    densities_str = "_".join(map(str, DENSITY_SORT_ORDER)).replace(".", "p")

    print(f"--- Starting Comparison for {workload} across densities: {', '.join(map(str, DENSITY_SORT_ORDER))} ---")

    results = defaultdict(lambda: defaultdict(dict))
    base_layer_names_set = set()
    all_layer_dirs_found = {}

    for density_val, density_dir_name in TARGET_DENSITIES.items():
        print(f"\nProcessing density: {density_val} ({density_dir_name})")
        model_density_dir = os.path.join(base_output_dir, workload, density_dir_name)

        if not os.path.isdir(model_density_dir):
            print(f"  Warning: Directory not found for density {density_val}: {model_density_dir}")
            all_layer_dirs_found[density_val] = []
            continue

        try:
            layer_dirs = [d for d in os.listdir(model_density_dir) if os.path.isdir(os.path.join(model_density_dir, d))]
            all_layer_dirs_found[density_val] = sorted(layer_dirs)
        except OSError as e:
             print(f"  ERROR: Could not list directory contents: {model_density_dir} - {e}")
             all_layer_dirs_found[density_val] = []
             continue

        for layer_config_name_with_wd in all_layer_dirs_found[density_val]:
            #Extract Base Layer Name
            match = re.match(r"^(.*)-WD\d+\.\d+", layer_config_name_with_wd)
            if match:
                base_layer_name = match.group(1)
            else:
                 print(f"    Warning: Could not extract base name from '{layer_config_name_with_wd}'. Using full name.")
                 base_layer_name = layer_config_name_with_wd

            base_layer_names_set.add(base_layer_name)

            layer_path = os.path.join(model_density_dir, layer_config_name_with_wd)

            for scheme_short_name, scheme_full_name in TARGET_SCHEMES.items():
                stats_file = os.path.join(layer_path, scheme_full_name, "output", STATS_FILENAME)
                parsed_data = parse_stats_txt(stats_file) 

                if parsed_data:
                    results[base_layer_name][density_val][scheme_short_name] = parsed_data
                else:
                    results[base_layer_name][density_val][scheme_short_name] = {'cycles': None, 'energy_pj': None}

    print("\nProcessing collected data into DataFrame...")
    all_data_list = []
    sorted_unique_layers = sorted(list(base_layer_names_set))

    for layer_config in sorted_unique_layers:
        for density_val in DENSITY_SORT_ORDER:
            for scheme_name in SCHEME_SORT_ORDER:
                density_data = results.get(layer_config, {}).get(density_val, {})
                data = density_data.get(scheme_name, {'cycles': None, 'energy_pj': None})

                all_data_list.append({
                    'layer_config': layer_config,
                    'density': density_val,
                    'scheme': scheme_name,
                    'cycles': data.get('cycles'),
                    'energy_pj': data.get('energy_pj')
                })

    if not all_data_list:
        print("ERROR: No data collected (list is empty). Cannot generate plot or table.")
        sys.exit(1)

    df = pd.DataFrame(all_data_list)

    df['cycles'] = pd.to_numeric(df['cycles'])
    df['energy_pj'] = pd.to_numeric(df['energy_pj'])

    print("\n--- DataFrame unique layer_config values before plotting ---")
    if not df.empty:
        print(df['layer_config'].unique())
    else:
        print("DataFrame is empty, cannot show unique layers.")
    print("----------------------------------------------------------")


    table_filename = os.path.join(base_output_dir, f"{workload}_{densities_str}_stc_dstc_comparison.csv")
    print(f"\nSaving combined results table to: {table_filename}")
    try:
        df_sorted_save = df.sort_values(by=['layer_config', 'density', 'scheme']).reset_index(drop=True)
        df_sorted_save.to_csv(table_filename, index=False, float_format='%.4f')
        print("Table saved successfully.")
    except Exception as e:
        print(f"ERROR: Could not save table to CSV: {e}")

    print("\nGenerating individual plots per layer configuration...")
    DSTC_COLOR = 'firebrick'
    STC_COLOR = 'cornflowerblue'
    scheme_colors = {'STC': STC_COLOR, 'DSTC': DSTC_COLOR}

    workload_plot_dir = os.path.join(base_output_dir, workload, 'plots')
    os.makedirs(workload_plot_dir, exist_ok=True)

    grouped_by_layer = df.groupby('layer_config')

    for layer_name, layer_df in grouped_by_layer:
        layer_plot_filename = os.path.join(workload_plot_dir, f"{layer_name}_stc_dstc_comparison.png")

        if layer_df.empty or layer_df[['cycles', 'energy_pj']].isnull().all().all():
             print(f"    Skipping layer {layer_name} due to missing data.")
             continue

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except IOError:
            try:
                plt.style.use('seaborn-whitegrid')
            except IOError:
                pass

        plt.rcParams.update({'font.size': 10})

        try:
             cycles_pivot = layer_df.pivot(index='density', columns='scheme', values='cycles').reindex(DENSITY_SORT_ORDER)
             energy_pivot = layer_df.pivot(index='density', columns='scheme', values='energy_pj').reindex(DENSITY_SORT_ORDER)
        except Exception as e:
            print(f"    ERROR pivoting data for layer {layer_name}: {e}. Skipping plot.")
            continue

        N_densities = len(DENSITY_SORT_ORDER)
        ind = np.arange(N_densities)
        bar_width = 0.35

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for i, scheme in enumerate(SCHEME_SORT_ORDER):
             offset = bar_width / 2 * (-1 if scheme == 'STC' else 1)
             color = scheme_colors[scheme]
             data_values = cycles_pivot[scheme].values if scheme in cycles_pivot.columns else [np.nan] * N_densities
             ax1.bar(ind + offset, data_values, bar_width, label=scheme, color=color)

        ax1.set_ylabel('Total Cycles', fontsize=12)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax1.legend(loc='upper right')
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        for i, scheme in enumerate(SCHEME_SORT_ORDER):
            offset = bar_width / 2 * (-1 if scheme == 'STC' else 1)
            color = scheme_colors[scheme]
            data_values = energy_pivot[scheme].values if scheme in energy_pivot.columns else [np.nan] * N_densities
            ax2.bar(ind + offset, data_values, bar_width, label=scheme, color=color)

        ax2.set_ylabel('Total Energy (pJ)', fontsize=12)
        ax2.set_ylim(bottom=0)
        ax2.set_xticks(ind)
        density_labels = [f'{d:.4f}' for d in DENSITY_SORT_ORDER]
        ax2.set_xticklabels(density_labels, rotation=45, ha='right')
        ax2.set_xlabel('Density', labelpad=5, fontsize=12)
        ax2.legend(loc='upper right')
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        fig.suptitle(f'DSTC vs. STC for {workload}\nLayer: {layer_name}', fontsize=14, y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        try:
             plt.savefig(layer_plot_filename, bbox_inches='tight')
        except Exception as e:
             print(f"    ERROR saving plot for layer {layer_name}: {e}")

        plt.close(fig)

    print("\nIndividual plot generation complete.")



if __name__ == "__main__":
    main()