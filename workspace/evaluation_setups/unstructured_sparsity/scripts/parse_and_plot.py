#!/usr/bin/env python3

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_stats_file(stats_file):
    if not os.path.exists(stats_file):
        print(f"Warning: Stats file not found: {stats_file}")
        return None

    stats = {'level_energy': {}}
    try:
        with open(stats_file, 'r') as f:
            content = f.read()

            #Extract total energy from Summary Stats
            summary_energy_match = re.search(r'Summary Stats\n-{3,}\n.*?Energy:\s+([\d.]+)\s+uJ', content, re.DOTALL)
            if summary_energy_match:
                stats['energy'] = float(summary_energy_match.group(1)) * 1_000_000 #uJ to pJ
            else:
                energy_match = re.search(r'Energy \(total\)\s+:\s+([\d.]+)', content)
                if energy_match:
                    print(f"Warning: Using fallback total energy parsing for {stats_file}")
                    stats['energy'] = float(energy_match.group(1))
                else:
                    print(f"Error: Could not parse total energy from {stats_file}")
                    stats['energy'] = 0

            #Extract cycles
            summary_cycles_match = re.search(r'Summary Stats\n-{3,}\n.*?Cycles:\s+(\d+)', content, re.DOTALL)
            if summary_cycles_match:
                stats['cycles'] = int(summary_cycles_match.group(1))
            else:
                cycles_match = re.search(r'Cycles\s+:\s+(\d+)', content)
                if cycles_match:
                     print(f"Warning: Using fallback cycles parsing for {stats_file}")
                     stats['cycles'] = int(cycles_match.group(1))
                else:
                     stats['cycles'] = 0


            #Extract utilization
            summary_util_match = re.search(r'Summary Stats\n-{3,}\n.*?Utilization:\s+([\d.]+)', content, re.DOTALL)
            if summary_util_match:
                stats['utilization'] = float(summary_util_match.group(1))
            else:
                 util_match = re.search(r'Utilization:\s+([\d.]+)', content)
                 if util_match:
                      print(f"Warning: Using fallback utilization parsing for {stats_file}")
                      try:
                         util_val = float(util_match.group(1))
                         stats['utilization'] = util_val / 100.0 if util_val > 1.0 else util_val
                      except ValueError:
                         stats['utilization'] = 0.0
                 else:
                     stats['utilization'] = 0.0

            level_pattern = r'(Level\s+(\d+)\n-{3,}\n=== ([^\n]+) ===.*?)(?=(Level\s+\d+\n-{3,})|\Z)'
            level_matches = re.finditer(level_pattern, content, re.DOTALL)

            level_energy = {}
            parsed_sum = 0.0
            for match in level_matches:
                level_full_header = match.group(1).strip()
                level_num = int(match.group(2))
                level_name = match.group(3).strip()
                level_content = match.group(1)

                energy_values = re.findall(r'Energy \(total\)\s+:\s+([\d.]+) pJ', level_content)
                total_level_energy = sum(float(e.replace(',', '')) for e in energy_values)

                level_key = f"L{level_num}_{level_name}"
                if total_level_energy > 0:
                     level_energy[level_key] = total_level_energy
                     parsed_sum += total_level_energy
                elif level_full_header:
                     level_energy[level_key] = 0.0

            stats['level_energy'] = level_energy

        if stats.get('energy', 0) > 0 and not np.isclose(parsed_sum, stats['energy'], rtol=0.1):
             print(f"Warning: Sum of parsed level energies ({parsed_sum:,.2f} pJ) doesn't closely match total summary energy ({stats.get('energy', 0):,.2f} pJ) for {stats_file}.")
        elif stats.get('energy', 0) == 0 and parsed_sum > 0:
             print(f"Warning: Total summary energy is 0, but parsed level energy sum is {parsed_sum:,.2f} pJ for {stats_file}.")

        return stats
    except Exception as e:
        print(f"Critical Error parsing {stats_file}: {e}")
        return None

def plot_comparison(all_stats, workloads, sparsity_types, figures_dir):
    """Plot grouped bar charts comparing sparsity types for cycles and energy."""

    display_workloads = [workload_display_names.get(w, w) for w in workloads]

    if not all_stats:
        print("No data available for plotting.")
        return

    num_workloads = len(workloads)
    num_sparsity_types = len(sparsity_types)
    x = np.arange(num_workloads)
    width = 0.8 / num_sparsity_types
    colors = ['red', 'blue']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.set_ylabel('Cycles')
    ax1.set_title('Cycles Comparison by Workload and Sparsity Type (at 50% Sparsity)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_workloads, rotation=45, ha="right")

    ax2.set_ylabel('Energy (pJ)')
    ax2.set_title('Energy Comparison by Workload and Sparsity Type (at 50% Sparsity)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_workloads, rotation=45, ha="right")

    sparsity_keys = list(all_stats.keys())

    for i, s_key in enumerate(sparsity_keys):
        cycles = [all_stats.get(s_key, {}).get(w, {}).get('cycles', 0) for w in workloads]
        energy = [all_stats.get(s_key, {}).get(w, {}).get('energy', 0) for w in workloads]

        offset = width * (i - (num_sparsity_types - 1) / 2)
        rects1 = ax1.bar(x + offset, cycles, width, label=s_key.replace('_', ' ').title(), color=colors[i % len(colors)])
        rects2 = ax2.bar(x + offset, energy, width, label=s_key.replace('_', ' ').title(), color=colors[i % len(colors)])

        for rect in rects1:
             height = rect.get_height()
             if height > 0:
                 ax1.text(rect.get_x() + rect.get_width()/2., height,
                         f'{int(height):,}', ha='center', va='bottom', fontsize=8)
        for rect in rects2:
            height = rect.get_height()
            if height > 0:
                ax2.text(rect.get_x() + rect.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=8)


    ax1.legend()
    ax2.legend()
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'sparsity_comparison.png'))
    plt.close()

    print(f"Comparison plot saved to {os.path.join(figures_dir, 'sparsity_comparison.png')}")

def create_dataframe(results, workloads):
    data = []
    for workload in workloads:
        if workload in results:
             display_name = workload_display_names.get(workload, workload)
             util_percent = results[workload].get('utilization', 0) * 100
             row = {
                'Workload': display_name,
                'Cycles': results[workload].get('cycles', 0),
                'Energy (pJ)': results[workload].get('energy', 0),
                'Utilization (%)': util_percent
            }
             data.append(row)
        else:
            display_name = workload_display_names.get(workload, workload)
            data.append({ 'Workload': display_name, 'Cycles': np.nan, 'Energy (pJ)': np.nan, 'Utilization (%)': np.nan})

    df = pd.DataFrame(data)
    df = df[['Workload', 'Cycles', 'Energy (pJ)', 'Utilization (%)']]
    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    output_dir = os.path.join(base_dir, 'outputs')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    workloads = ["resnet50_conv1", "alexnet_conv1", "mobilenet_conv1"]
    workload_display_names = {
        'resnet50_conv1': 'ResNet-50 C1',
        'alexnet_conv1': 'AlexNet C1',
        'mobilenet_conv1': 'MobileNetV1 C1'
    }

    unstructured_results = {}
    structured_results = {}

    print("Parsing results...")
    for workload in workloads:
        unstructured_stats_file = os.path.join(output_dir, f"unstructured_{workload}", "timeloop-mapper.stats.txt")
        stats = parse_stats_file(unstructured_stats_file)
        if stats:
            unstructured_results[workload] = stats

        structured_stats_file = os.path.join(output_dir, f"structured_{workload}", "timeloop-mapper.stats.txt")
        stats = parse_stats_file(structured_stats_file)
        if stats:
            structured_results[workload] = stats

    all_stats = {
        'Unstructured': unstructured_results,
        'Structured': structured_results
    }

    unstructured_df = create_dataframe(unstructured_results, workloads)
    structured_df = create_dataframe(structured_results, workloads)

    if not unstructured_df.empty and not structured_df.empty:
        comparison_df = pd.merge(unstructured_df, structured_df, on='Workload', suffixes=('_unstr', '_str'), how='outer')

        comparison_df['Cycle Ratio (Unstructured/Structured)'] = comparison_df['Cycles_unstr'] / comparison_df['Cycles_str'].replace(0, np.nan)
        comparison_df['Energy Ratio (Unstructured/Structured)'] = comparison_df['Energy (pJ)_unstr'] / comparison_df['Energy (pJ)_str'].replace(0, np.nan)
        comparison_df['Utilization Ratio (Structured/Unstructured)'] = comparison_df['Utilization (%)_str'] / comparison_df['Utilization (%)_unstr'].replace(0, np.nan)

        comparison_df = comparison_df[['Workload', 'Cycle Ratio (Unstructured/Structured)',
                                       'Energy Ratio (Unstructured/Structured)', 'Utilization Ratio (Structured/Unstructured)']]
    else:
        comparison_df = pd.DataFrame(columns=['Workload', 'Cycle Ratio (Unstructured/Structured)',
                                              'Energy Ratio (Unstructured/Structured)', 'Utilization Ratio (Structured/Unstructured)'])

    print("\nEnergy Breakdown by Hardware Level:")
    print("=" * 60)
    for arch_name, results in all_stats.items():
        print(f"\n{arch_name} Architecture:")

        available_workloads = [w for w in workloads if w in results]

        if not available_workloads:
            print("No results parsed for this architecture.")
            continue

        for workload in available_workloads:
            display_name = workload_display_names.get(workload, workload)
            if 'level_energy' in results[workload]:
                level_energy = results[workload]['level_energy']
                total_energy = results[workload].get('energy', 0)

                if level_energy and total_energy > 0:
                    print(f"\n  {display_name}:")
                    sorted_levels = sorted(level_energy.items(), key=lambda item: item[0])
                    for level, energy in sorted_levels:
                        percentage = (energy / total_energy) * 100
                        print(f"    {level}: {energy:,.2f} pJ ({percentage:.2f}%)")
                else:
                    print(f"\n  {display_name}: Energy breakdown not available or total energy is zero.")
            else:
                 print(f"\n  {display_name}: Parsed stats missing 'level_energy' key.")

    plot_comparison(all_stats, workloads, list(all_stats.keys()), figures_dir) 