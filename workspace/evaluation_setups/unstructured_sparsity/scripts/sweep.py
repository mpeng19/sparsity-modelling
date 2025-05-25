import argparse
import os
import shutil
import subprocess
import sys
import yaml
from pathlib import Path

def merge_yamls(file_paths):
    """Loads multiple YAML files and merges them into a single dictionary."""
    merged_data = {}
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    merged_data.update(data)
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}", file=sys.stderr)
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {file_path}: {e}", file=sys.stderr)
            sys.exit(1)
    return merged_data

def run_timeloop(command_args, cwd):
    """Runs a Timeloop command using subprocess."""
    print(f"Running command: {' '.join(map(str, command_args))}")
    try:
        process = subprocess.run(
            command_args,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        print("Timeloop stdout:")
        print(process.stdout)
        print("Timeloop stderr:")
        print(process.stderr)
        return True
    except FileNotFoundError:
         print(f"Error: 'timeloop-mapper' command not found. Make sure Timeloop is installed and in your PATH.", file=sys.stderr)
         sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running Timeloop: Exit code {e.returncode}", file=sys.stderr)
        print(f"Stdout:\n{e.stdout}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Timeloop mapper for specified mode and workloads.")
    parser.add_argument(
        "mode",
        choices=["unstructured", "structured"],
        help="The evaluation mode: 'unstructured' or 'structured'."
    )
    args = parser.parse_args()
    mode = args.mode

    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent
    output_base = base_dir / "outputs"
    mappings_base = base_dir / "mappings_found"
    workload_dir = base_dir / "workload"
    sparse_opt_dir = base_dir / "sparse-opt"

    output_base.mkdir(parents=True, exist_ok=True)
    mappings_base.mkdir(parents=True, exist_ok=True)

    if mode == "unstructured":
        arch_file = base_dir / "arch" / "unstructured_sparse_tensor_core.yaml"
        dataflow_file = base_dir / "dataflow" / "unstructured_dataflow.yaml"
        sparse_opt_file = sparse_opt_dir / "unstructured.yaml"
        ert_file = base_dir / "ert_art" / "ERT.yaml"
        art_file = base_dir / "ert_art" / "ART.yaml"
    elif mode == "structured":
        arch_file = base_dir / "arch" / "structured_2_4_tensor_core.yaml"
        dataflow_file = base_dir / "dataflow" / "structured_2_4_dataflow.yaml"
        sparse_opt_file = sparse_opt_dir / "structured_2_4.yaml"
        ert_file = base_dir / "ert_art" / "ERT_structured.yaml"
        art_file = base_dir / "ert_art" / "ART_structured.yaml"
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    mapper_file = base_dir / "mapper" / "mapper.yaml"
    output_suffix = mode
    stored_map_dir = mappings_base / output_suffix
    stored_map_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Mode: {mode} ===")
    print(f"Arch: {arch_file}")
    print(f"Dataflow: {dataflow_file}")
    print(f"SparseOpt: {sparse_opt_file}")
    print(f"ERT: {ert_file}")
    print(f"ART: {art_file}")
    print(f"Mapper: {mapper_file}")
    print()

    workloads = ["resnet50_conv1", "alexnet_conv1", "mobilenet_conv1"]

    for w in workloads:
        print(f"--- {w} ({mode}) ---")
        workload_file = workload_dir / f"{w}.yaml"
        if not workload_file.is_file():
            print(f"Workload file {workload_file} not found. Skipping.", file=sys.stderr)
            continue

        tmp_out_dir = output_base / f"_tmp_{mode}_{w}"
        final_out_dir = output_base / f"{mode}_{w}"
        stored_map_file = stored_map_dir / f"{w}.map.yaml"

        tmp_out_dir.mkdir(parents=True, exist_ok=True)
        final_out_dir.mkdir(parents=True, exist_ok=True)

        print("--- Stage 1: Mapping Search ---")
        aggregated_input_file = tmp_out_dir / "aggregated_inputs.yaml"
        map_out_file = tmp_out_dir / "timeloop-mapper.map.yaml"

        input_files_stage1 = [
            arch_file,
            dataflow_file,
            sparse_opt_file,
            workload_file,
            mapper_file,
        ]
        aggregated_data = merge_yamls(input_files_stage1)

        try:
            with open(aggregated_input_file, 'w') as f:
                yaml.dump(aggregated_data, f, default_flow_style=False)
        except IOError as e:
            print(f"Error writing aggregated input file {aggregated_input_file}: {e}", file=sys.stderr)
            shutil.rmtree(tmp_out_dir)
            continue

        cmd_stage1 = [
            "timeloop-mapper",
            aggregated_input_file,
            ert_file,
            art_file,
            "-o", "."
        ]

        if not run_timeloop(cmd_stage1, cwd=tmp_out_dir):
            print(f"Mapping search failed for {w}. Skipping.", file=sys.stderr)
            shutil.rmtree(tmp_out_dir)
            continue

        if not map_out_file.is_file():
            print(f"Mapping search completed but map file {map_out_file} not found for {w}. Skipping.", file=sys.stderr)
            shutil.rmtree(tmp_out_dir)
            continue

        try:
            shutil.copy(map_out_file, stored_map_file)
            print(f"Mapping file saved to: {stored_map_file}")
        except Exception as e:
            print(f"Error copying map file {map_out_file} to {stored_map_file}: {e}", file=sys.stderr)
            shutil.rmtree(tmp_out_dir)
            continue

        shutil.rmtree(tmp_out_dir)

        print("--- Stage 2: Model Evaluation ---")

        cmd_stage2 = [
            "timeloop-mapper",
            arch_file,
            dataflow_file,
            sparse_opt_file,
            workload_file,
            mapper_file,
            stored_map_file,
            ert_file,
            art_file,
            "-o", "."
        ]

        if not run_timeloop(cmd_stage2, cwd=final_out_dir):
             print(f"Model evaluation failed for {w}.", file=sys.stderr)
        else:
            print(f"Results in {final_out_dir}")

    print("Done.")

if __name__ == "__main__":
    main() 