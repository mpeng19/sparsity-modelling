import re
import sys
import os

def parse_stats_txt(stats_file_path):
    """
    Parses a timeloop-model.stats.txt file to extract Cycles and Total Energy.
    Cycles are read from the MAC level.
    Energy is summed from the 'Energy (total)' line of each level.

    Args:
        stats_file_path (str): The path to the .stats.txt file.

    Returns:
        dict: A dictionary containing {'cycles': int, 'energy_pj': float}
              Returns None if the file doesn't exist or parsing fails.
              Energy is returned in picoJoules (pJ).
    """
    if not os.path.exists(stats_file_path):
        # print(f"Warning: Stats file not found: {stats_file_path}")
        return None

    stats = {'cycles': None, 'energy_pj': 0.0} # Initialize energy to 0.0 for summing
    found_energy = False
    found_cycles = False

    try:
        with open(stats_file_path, 'r') as f:
            content = f.read()

            # --- Extract Cycles ---
            # Look specifically under the "=== MAC ===" section
            mac_section_match = re.search(r'===\s*MAC\s*===(.*?)Level \d+', content, re.DOTALL | re.IGNORECASE)
            if mac_section_match:
                mac_content = mac_section_match.group(1)
                # Find Cycles within the MAC section
                cycles_match = re.search(r'^\s*Cycles\s*:\s*(\d+)', mac_content, re.MULTILINE)
                if cycles_match:
                    stats['cycles'] = int(cycles_match.group(1))
                    found_cycles = True
                else:
                     print(f"Warning: Could not find 'Cycles:' within MAC section in {stats_file_path}")
            else:
                 print(f"Warning: Could not find '=== MAC ===' section in {stats_file_path}")

            # --- Extract and Sum Energy ---
            # Find all occurrences of 'Energy (total)' lines with pJ units
            energy_matches = re.findall(r'^\s*Energy\s*\(total\)\s*:\s*([\d.]+)\s*pJ', content, re.MULTILINE | re.IGNORECASE)

            if energy_matches:
                for energy_str in energy_matches:
                    try:
                        stats['energy_pj'] += float(energy_str)
                        found_energy = True
                    except ValueError:
                         print(f"Warning: Could not convert energy value '{energy_str}' to float in {stats_file_path}")
            else:
                 print(f"Warning: Could not find any 'Energy (total): ... pJ' lines in {stats_file_path}")


    except Exception as e:
        print(f"Error parsing file {stats_file_path}: {e}")
        return None

    # Check if we found the essential stats
    if not found_cycles or not found_energy:
        print(f"Warning: Failed to extract required cycles and/or energy data from {stats_file_path}")
        # Return None only if cycles are missing, allow partial energy sum if cycles found
        if not found_cycles:
             return None
        # If only energy is missing, we might still proceed but energy will be 0.0
        # Resetting energy to None if it remained 0.0 and wasn't truly found
        if not found_energy:
             stats['energy_pj'] = None
             return None # Treat missing energy as critical failure as well


    return stats

if __name__ == '__main__':
    # Example Usage:
    if len(sys.argv) < 2:
        print("Usage: python parse_timeloop_stats_file.py <path_to_stats_file>")
        # Provide a default path for testing if needed
        # test_path = "../outputs/resnet50_selected/WD-0.5/M128-K1152-N1024-IAD0.44-WD0.5/DSTC-RF2x-24-bandwidth/output/timeloop-model.stats.txt"
        # print(f"Running with test path: {test_path}")
        # if os.path.exists(test_path):
        #      parsed_data = parse_stats_txt(test_path)
        # else:
        #      print(f"Test path not found: {test_path}")
        #      parsed_data = None
        sys.exit(1)
    else:
        file_path = sys.argv[1]
        parsed_data = parse_stats_txt(file_path)

    if parsed_data:
        print(f"Parsed data for {file_path}:")
        print(f"  Cycles: {parsed_data.get('cycles', 'Not found')}")
        print(f"  Energy (pJ): {parsed_data.get('energy_pj', 'Not found')}")
    else:
        print(f"Could not parse required stats from {file_path}") 