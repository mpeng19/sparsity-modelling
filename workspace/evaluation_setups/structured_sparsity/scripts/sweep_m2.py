import os, inspect, sys, subprocess, yaml, pprint, math, pickle, shutil, signal, math, time, datetime
from copy import deepcopy
import argparse


this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
os.chdir(this_directory)

OVERWRITE  = True
VERBOSE    = False
PRECISION  = "int8"
USE_MODEL  = False
MAX_LAYERS = 100


def run_timeloop(job_name, input_dict, ert_path, art_path, base_dir):
    output_dir = os.path.join(base_dir +"/output")
    
    if not os.path.exists(output_dir) and not os.path.exists(os.path.join(output_dir, "timeloop-mapper.map+stats.xml")) and not os.path.exists(os.path.join(output_dir, "timeloop-model.map+stats.xml")):
       os.makedirs(output_dir)
    else:
        if not OVERWRITE:
            print("Found existing results: ", output_dir)
            return True
        else:
            print("Found and overwrite existing results: ", output_dir)
    
    input_file_path = os.path.join(base_dir, "aggregated_input.yaml")
    shutil.copy(ert_path, os.path.join(base_dir, "ERT.yaml"))
    shutil.copy(art_path, os.path.join(base_dir, "ART.yaml"))
    logfile_path = os.path.join(output_dir, "timeloop.log")
    
    if not USE_MODEL: 
        yaml.dump(input_dict, open(input_file_path, "w"), default_flow_style=False)
        os.chdir(output_dir)
        subprocess_cmd = ["timeloop-mapper", input_file_path, os.path.join(base_dir, "ERT.yaml"), os.path.join(base_dir, "ART.yaml")]

        # print("\tRunning test: ", job_name)

        p = subprocess.Popen(subprocess_cmd)
        try:
            p.communicate(timeout=1200) # wait for at most 20 min
        except KeyboardInterrupt:
            print(job_name,  " manually stopped")
            os.kill(p.pid, signal.SIGINT)
            pass
        except subprocess.TimeoutExpired:
            print(job_name,  " reaches timeout limit")
            os.kill(p.pid, signal.SIGINT)
        
        counter = 0
        while counter <= 100 and not os.path.exists(os.path.join(output_dir, "timeloop-mapper.map+stats.xml")):
            time.sleep(1)
            counter = counter + 1
            if counter == 100:
                print("cannot find output after 100 sec, exiting...")
                print("output causing this issue is: ", os.path.join(output_dir, "timeloop-mapper.map+stats.xml"))
                sys.exit(1)  

        outfile = os.path.join(output_dir, "timeloop-mapper.map.yaml")
        success = os.path.exists(outfile)

        if success:
            path_elements = base_dir.split(os.sep)
            design_name = path_elements[-1]
            
            try: 
                mappings_found_base = os.path.abspath(os.path.join(this_directory, os.pardir, "mappings_found"))
                
                target_mapping_dir = os.path.join(mappings_found_base, PRECISION, 
                                                  path_elements[-4], # workload
                                                  path_elements[-3], # WD-degree
                                                  path_elements[-2], # instance_name
                                                  design_name)       # design_name
                target_mapping_path = os.path.join(target_mapping_dir, "map.yaml")
                
                print(f"INFO: Mapper success for {job_name}. Storing map...")
                print(f"      Source: {outfile}")
                print(f"      Target: {target_mapping_path}")
                os.makedirs(target_mapping_dir, exist_ok=True)
                shutil.copy(outfile, target_mapping_path)
            except Exception as e:
                print(f"WARN: Failed to auto-store mapping file for {job_name}.")
                print(f"      Base dir structure might be unexpected: {base_dir}")
                print(f"      Path elements: {path_elements}")
                print(f"      Error: {e}")

        return True
    else:
        
        model_input_dict = deepcopy(input_dict)
        model_input_dict.pop("architecture_constraints")
        model_input_dict.pop("mapper")
        yaml.dump(model_input_dict, open(input_file_path, "w"), default_flow_style=False)
        
        path_elements = base_dir.split(os.sep)
        design_name = path_elements[-1]
        mapping_path = os.path.join(this_directory, os.pardir, "mappings_found", PRECISION, path_elements[-4], path_elements[-3], path_elements[-2], design_name, "map.yaml")
        
        if not (os.path.exists(mapping_path)):
            print("ERROR: cannot find mapping: ", mapping_path)
            exit(1)
        
        shutil.copy(mapping_path, os.path.join(base_dir, "map.yaml"))
        
        os.chdir(output_dir)
        subprocess_cmd = ["timeloop-model", input_file_path, os.path.join(base_dir, "ERT.yaml"), os.path.join(base_dir, "ART.yaml"), os.path.join(base_dir, "map.yaml")]
        p = subprocess.Popen(subprocess_cmd)
        
        counter = 0
        while not os.path.exists(os.path.join(output_dir, "timeloop-model.map+stats.xml")):
            counter = counter + 1
            if counter == 500:
                print("cannot find output after 500 sec, exiting...")
                print("output causing this issue is: ", os.path.join(output_dir, "timeloop-model.map+stats.xml"))
                print("if your machine can be slow, update the timeout setup at: ", this_file_path)
                sys.exit(1)
            time.sleep(1)
            
        return True

def process_constraints(design_point, aggregated_input):

    if "-RF1x" in design_point["design_name"]:
        RF_case = "RF1x" 
    elif "-RF2x" in design_point["design_name"]:
        RF_case = "RF2x"
    else:
        RF_case = "RF4x"
    
    dataflow_case = "Ms" if "Ms" in design_point["configurations"]["dataflow"] else "Os"
    if RF_case == "RF2x" and "Ms" in dataflow_case:
        architecture_constraints = deepcopy(aggregated_input["architecture_constraints"])
        for target in architecture_constraints["targets"]:
            if target["target"] == "LRF" and target["type"] == "temporal":
                target["factors"] = "K=1 M=1 N<=128"
        aggregated_input["architecture_constraints"] = architecture_constraints
    elif dataflow_case == "Os":
        pass
    else:
        assert(False);
    
    if "representation" not in design_point["configurations"] or design_point["configurations"]["representation"] == "U":
        return aggregated_input

    # further shrink mapspace for some of the cases
    sparsity_scheme_case = design_point["configurations"]["sparsity_schemes"]["W"]
    if sparsity_scheme_case in ["2-8"] and "Ms" in dataflow_case:
        for target in architecture_constraints["targets"]:
            if target["target"] == "RF" and target["type"] == "spatial":
                target["factors"] = "K=64 M=16 N=1"   
    elif sparsity_scheme_case in ["2-6"] and "Ms" in dataflow_case:
        for target in architecture_constraints["targets"]:
            if target["target"] == "RF" and target["type"] == "spatial":
                target["factors"] = "K=48 M=16 N=1"   
    elif sparsity_scheme_case in ["2-4"] and "Ms" in dataflow_case:
        for target in architecture_constraints["targets"]:
            if target["target"] == "RF" and target["type"] == "spatial":
                target["factors"] = "K=32 M=16 N=1"      

    return aggregated_input

def process_format(design_point, aggregated_input):
    if "representation" not in design_point["configurations"] or design_point["configurations"]["representation"] == "U":
        return aggregated_input
    
    repr_case = design_point["configurations"]["representation"]
    sparsity_scheme_case = design_point["configurations"]["sparsity_schemes"]["W"]
    
    # Remove weight compression if the weights are dense
    # applicable to any architecture
    if sparsity_scheme_case == "dense" or sparsity_scheme_case == "1.0":
        for target in aggregated_input["sparse_optimizations"]["targets"]:
            if "representation-format" in target:
                new_repr = {"data-spaces":[]}
                for item in target["representation-format"]["data-spaces"]:
                    if item["name"] == "A":
                        continue
                    else:
                        new_repr["data-spaces"].append(item)
                target["representation-format"] = new_repr 
        return aggregated_input
    
    if repr_case == "CP":
        if sparsity_scheme_case == "2-4":
            num_metadata_bits = 2;
        elif sparsity_scheme_case in ["2-6", "2-8"]:
            num_metadata_bits = 3
        else:
            print(sparsity_scheme_case)
            assert(False)
        
        for target in aggregated_input["sparse_optimizations"]["targets"]:
            if "representation-format" in target:
                target["representation-format"]["data-spaces"][0]["ranks"][-1]["metadata-word-bits"] = num_metadata_bits
                target["representation-format"]["data-spaces"][0]["ranks"][-1]["format"] = repr_case
    
    elif repr_case == "RLE":
        if sparsity_scheme_case == "2-6":
            num_metadata_bits = 2
        elif sparsity_scheme_case == "2-4":
            num_metadata_bits = 2
        elif sparsity_scheme_case == "2-8":
            num_metadata_bits = 3
        else:
            assert(False)
            
        
        for target in aggregated_input["sparse_optimizations"]["targets"]:
            if "representation-format" in target:
                target["representation-format"]["data-spaces"][0]["ranks"][-1]["metadata-word-bits"] = num_metadata_bits
                target["representation-format"]["data-spaces"][0]["ranks"][-1]["format"] = repr_case
     
    elif repr_case == "B":
        num_l0_metadata_bits = 1
        num_l1_metadata_bits = 0;
        for target in aggregated_input["sparse_optimizations"]["targets"]:
            if "representation-format" in target and target["representation-format"]["data-spaces"][0]["name"]=="A":
                try:
                    target["representation-format"]["data-spaces"][0]["ranks"][-1]["metadata-word-bits"] = num_l0_metadata_bits
                    target["representation-format"]["data-spaces"][0]["ranks"][-1]["format"] = repr_case
                except:
                    print("Warn: cannot apply bitmask metadata-word-bits updates")
                    pass
    else:
        assert(False)
    
    return aggregated_input

def process_instance_dimension(design_point, aggregated_input):
    
    #pprint.pprint(design_point)
    
    if not  "sparsity_schemes" in design_point["configurations"]:
        return aggregated_input

    sparsity_scheme = design_point["configurations"]["sparsity_schemes"]["W"]
    
    #pad the K dimension such that the structure is well maintained in the workload
    if sparsity_scheme == "dense":
        return aggregated_input
    
    if sparsity_scheme in ["2-4"]:
        min_granularity = 4
    elif sparsity_scheme in ["2-6", "A-U0.3333"]:
        min_granularity = 48
    elif sparsity_scheme in ["2-8"]:
        min_granularity = 8
    elif "0.5" in sparsity_scheme:
        min_granularity = 2
    elif "0.25" in sparsity_scheme:
        min_granularity = 4
    elif "0.375" in sparsity_scheme:
        min_granularity = 8
    else:
        print("ERROR: scheme not recognized: ", sparsity_scheme)
        sys.exit(1)
   
    orig_K = aggregated_input["problem"]["instance"]["K"]
    padded_K = orig_K
    if not orig_K % min_granularity == 0:
        padded_K = math.ceil(orig_K/min_granularity)*min_granularity
    aggregated_input["problem"]["instance"]["K"] = padded_K
    
    if VERBOSE:
        if not padded_K == orig_K:
            print("padding applied to maintain structure: padded K: ", padded_K, "  orig K: ", orig_K)
    
    #2-6 needs special padding
    if design_point["configurations"]["sparsity_schemes"] == "2-6":
        ratio = 6/2
        min_spatial_K = int(16*ratio)
        K = aggregated_input["problem"]["instance"]["K"]
        aggregated_input["problem"]["instance"]["K"] = int(math.ceil(K/math.ceil(min_spatial_K))*min_spatial_K)

    return aggregated_input


def hw_design_point_run(raw_input_templates, base_dir):
   
    input_templates = deepcopy(raw_input_templates) 
    aggregated_input = {}
    aggregated_input["problem"] = {}
    aggregated_input["problem"].update(input_templates["prob_instance"][1])
    aggregated_input["problem"].update(yaml.load(open(os.environ.get("gemm-ABZ")), Loader=yaml.SafeLoader))
    aggregated_input["problem"]["instance"]["densities"] = deepcopy(input_templates["sparsity_schemes"]["densities"])
    
    sparsity_scheme = "dense"
    if "sparse_optimizations" in input_templates:
        sparsity_scheme = input_templates["design_point"]["configurations"]["sparsity_schemes"]["W"]
        try:
            aggregated_input.update(input_templates["sparse_optimizations"])
        except:
            pprint.pprint(input_templates)

    data_representation = "U"
    if "representation" in input_templates:
        data_representation = input_templates["representation"]
    
    aggregated_input.update(input_templates["compound_components"])
    aggregated_input.update(input_templates["design"])
    aggregated_input.update(input_templates["dataflow"])
    aggregated_input.update(input_templates["mapper"])
    
    aggregated_input = process_constraints(input_templates["design_point"], aggregated_input)
    aggregated_input = process_format(input_templates["design_point"], aggregated_input)
    aggregated_input = process_instance_dimension(input_templates["design_point"], aggregated_input)
    problem_name = input_templates["prob_instance"][0]
    
    dataflow_case = "Ms" if "Ms" in input_templates["design_point"]["configurations"]["dataflow"] else "Os"
    
    design_point = input_templates["design_point"]["design_name"] + "_" \
                   + dataflow_case + "_" \
                   + sparsity_scheme + "_" + data_representation
            
    
    # print("\nInfo: output will be saved to: ", base_dir)
   
    # get the ert and art paths
    template_root = os.environ.get("SYSTEM_TEMPLATES_ROOT")
    design_ert_art_base_dir = os.path.join(template_root, "ert_art", PRECISION, input_templates["design_point"]["design_name"])
    ert_path = os.path.join(design_ert_art_base_dir, "ERT.yaml")
    art_path = os.path.join(design_ert_art_base_dir, "ART.yaml")
    
    aggregated_input = allow_streaming(aggregated_input)
    success = run_timeloop(base_dir, aggregated_input, ert_path, art_path, base_dir)

    if (USE_MODEL and not success):
        print("ERROR: mapping illegal...")
        exit(1)

def allow_streaming(aggregated_input):
    processed_input = deepcopy(aggregated_input)
    assert(processed_input["architecture"]["subtree"][0]["subtree"][0]["local"][0]["name"] == "SMEM")
    
    for constriant in processed_input["architecture_constraints"]:
        for target in processed_input["architecture_constraints"]["targets"]:
            if target["target"] == "SMEM" and target["type"] == "bypass":
                target.pop("keep")
                target["stream"] = ["A", "B"]
    return processed_input 


def generate_prob_instances(pkl_path, max_layers, layer_index = None):
    
    prob_instances = pickle.load(open(pkl_path, "rb"))
    if layer_index is None:
        limit = max_layers
        instance_names = sorted(prob_instances)[0:limit] if limit <= len(prob_instances.keys()) else sorted(prob_instances)
        instance_specs = {}
        for instance_name in instance_names:
            instance_specs[instance_name] = prob_instances[instance_name]["timeloop_spec"]
    else:
        instance_names = sorted(prob_instances)
        assert(layer_index < len(instance_names))

        instance_name = instance_names[layer_index]
        instance_specs = {}
        instance_specs[instance_name] = prob_instances[instance_name]["timeloop_spec"]

    return instance_specs


def derive_density_scheme(degree_to_scheme, design_point):
    
    template_root = os.environ.get("SYSTEM_TEMPLATES_ROOT")
    schemes_path = os.path.join(template_root, "configurations", "sparsity_schemes")

    base_design = design_point["design_name"].split("-")[0]

    # experiment_key = design_point["design_name"]
    experiment_key = ""
 
    default_dense = yaml.load(open(os.path.join(schemes_path, "dense-1.0.yaml")), Loader = yaml.SafeLoader)
    design_point["processed_schemes"] = deepcopy(default_dense)
        
    WD_degree = float(design_point["workload"]["density degrees"]["A"])
    IAD_degree = float(design_point["workload"]["density degrees"]["B"])

    design_point["configurations"]["sparsity_schemes"] = {"W": "dense", "I": "dense"}
    
    #Generate the sparsity schemes for different architectures
    if WD_degree == 1.0 and IAD_degree == 1.0:
        if ("sparse_optimizations" in design_point["configurations"]):
            design_point["configurations"].pop("sparse_optimizations")
        
        if ("representation" in design_point["configurations"]):
            design_point["configurations"].pop("representation")
        
        experiment_key = experiment_key + "WDU1.0-IADU1.0"
    
    else:
        if WD_degree != 1.0:
            base_design = design_point["design_name"].split("-")[0]
            weight_scheme = degree_to_scheme[base_design]["WD"][WD_degree]
            scheme_spec = yaml.load(open(os.path.join(schemes_path, weight_scheme + ".yaml")), Loader = yaml.SafeLoader)
            experiment_key = experiment_key + "WD" + weight_scheme 
            design_point["configurations"]["sparsity_schemes"]["W"] = weight_scheme
           
            # get that scheme
            try:
                design_point["processed_schemes"]["densities"]["A"] = deepcopy(scheme_spec["A"])
            except:
                design_point["processed_schemes"]["densities"]["A"] = deepcopy(scheme_spec["B"])
        else:
            experiment_key = experiment_key + "-WDU1.0"
            design_point["configurations"]["sparsity_schemes"]["W"] = "dense"
        
        if IAD_degree != 1.0:
        
            IAD_granularity = degree_to_scheme[base_design]["IAD"]["granularity"]
            if IAD_granularity == "dense":
                experiment_key = experiment_key + "-IADU1.0"
                pass
            elif IAD_granularity == "any":
                design_point["processed_schemes"]["densities"]["B"]["distribution"] = "hypergeometric"
                design_point["processed_schemes"]["densities"]["B"]["density"] = IAD_degree
                experiment_key = experiment_key + "-IADU" + str(IAD_degree)
                design_point["configurations"]["sparsity_schemes"]["I"] = "U" + str(IAD_degree)
            elif IAD_granularity == 0.125:
                assert("S2TA" in design_point["design_name"])
                n = math.ceil(IAD_degree/0.125)
                if n <= 5:
                    new_IAD = n * 0.125
                    design_point["processed_schemes"]["densities"]["B"]["density"] = new_IAD
                    experiment_key = experiment_key + "-IAD" + str(new_IAD)
                    design_point["configurations"]["sparsity_schemes"]["I"] = str(n) + ":8" 
                else:
                    new_IAD = IAD_degree
                    design_point["processed_schemes"]["densities"]["B"]["distribution"] = "hypergeometric"
                    design_point["processed_schemes"]["densities"]["B"]["density"] = IAD_degree
                    experiment_key = experiment_key + "-IAD" + str(new_IAD)
        else:
            experiment_key = experiment_key + "-IADU1.0"
    
    #Select correct architecture setup and/or sparse optimizations
    if "DSTC" in base_design:
        is_special_bw = "DSTC-RF2x-24-bandwidth" in design_point["design_name"] 
        if IAD_degree < 0.8 or WD_degree < 0.8:
            # we need the queues to handle bank conflicts, we are already being conservative here 
            # since original paper says only complete dense case does not use queues
            design_point["design_name"] = "DSTC-RF2x" 
            design_point["configurations"]["sparse_optimizations"] = "LRF-PosTiling"
        else:
            # we do not need the queues to handle bank conflicts
            design_point["design_name"] = "DSTC-RF2x-ideal-RF" 
            design_point["configurations"]["sparse_optimizations"] = "LRF-PosTiling"
        
        if is_special_bw:
            design_point["design_name"] = "DSTC-RF2x-24-bandwidth"
    
    elif "FlexibleGHCompressAB-RF2x-24-bandwidth" in design_point["design_name"]:
        if IAD_degree < 0.8:
            design_point["configurations"]["sparse_optimizations"] = "RF-spatial-skipping-compressAB"
        else:
            design_point["configurations"]["sparse_optimizations"] = "RF-spatial-skipping"
    else:
        # architecture does not need any processing
        pass   
    
    experiment_key = design_point["design_name"] + "-" + experiment_key
    design_point["experiment_key"] = experiment_key
    
    return design_point


def top(workload, density_degrees, verbose, max_layers, layer_idx=None):
    if not os.path.exists(os.environ["RESULTS_ROOT"]):
        os.makedirs(os.environ["RESULTS_ROOT"])

    experiment_record = {}

    template_root      = os.environ["SYSTEM_TEMPLATES_ROOT"]
    compound_components = yaml.load(open(os.path.join(
        template_root, "../../components/compound_components.yaml")), Loader=yaml.SafeLoader)
    mapper             = yaml.load(open(os.path.join(
        template_root, "../../mapper/mapper.yaml")), Loader=yaml.SafeLoader)
    degree_to_scheme   = yaml.load(open(os.path.join(
        this_directory, "design-to-scheme.yaml")), Loader=yaml.SafeLoader)

    sweep_yaml = yaml.load(open(os.path.join(
        template_root, "design_points", sweep_name + ".yaml")), Loader=yaml.SafeLoader)

    #Filter to STC & DSTC only
    valid_designs = {"STC-RF2x-24-bandwidth", "DSTC-RF2x-24-bandwidth"}
    design_points = [d for d in sweep_yaml["design_points"]
                     if d["design_name"] in valid_designs]

    for WD_degree in density_degrees:
        relative_path = os.path.join("pkl_specs", f"{workload}_specs.{WD_degree}.pkl")
        path = os.path.join(os.environ.get("WORKLOADS"), relative_path)
        
        if not os.path.exists(path):
            print("ERROR: pkl file does not exist: ", path)
            sys.exit(1)

        print("\nINFO: generating per layer results for: ", WD_degree)
        
        instance_specs = generate_prob_instances(path, max_layers, layer_idx)
        for instance_name, instance_spec in instance_specs.items():
            IAD_degree = instance_name.split("-IAD")[1].split("-")[0]

            for dp in design_points:
                dp = deepcopy(dp)
                if dp["design_name"].startswith("DSTC"):
                   dp.setdefault("configurations", {})["representation"] = "U"
                if dp["design_name"].startswith("STC"):
                    dp.setdefault("configurations", {})["representation"] = "U"

                dp["workload"] = {"name": workload,
                                  "density degrees": {"A": WD_degree, "B": IAD_degree}}
                dp = derive_density_scheme(degree_to_scheme["designs"], dp)

                #Remove specific buggy sparse optimizations for STC/DSTC
                if dp["design_name"].startswith("DSTC") or dp["design_name"].startswith("STC"):
                    if "sparse_optimizations" in dp.get("configurations", {}):
                        if dp['configurations']['sparse_optimizations'] == "LRF-PosTiling":
                            dp["configurations"].pop("sparse_optimizations")
                        elif dp['configurations']['sparse_optimizations'] == "RF-spatial-skipping":
                            dp["configurations"].pop("sparse_optimizations")

                out_dir = os.path.join(os.environ["RESULTS_ROOT"],
                                       workload, f"WD-{WD_degree}", instance_name,
                                       dp["design_name"])
                parts = instance_name.split("-")
                exp_key = parts[1] + parts[2] + parts[3] + "  <-> " + dp["experiment_key"]
                if exp_key in experiment_record:
                    shutil.copytree(experiment_record[exp_key], out_dir, dirs_exist_ok=True)
                    continue

                input_templates = {
                    "compound_components": compound_components,
                    "mapper": mapper,
                    "prob_instance": [instance_name, instance_spec],
                    "design_point": deepcopy(dp),
                }

                arch_yaml = os.path.join(template_root, "designs", PRECISION,
                                         dp["design_name"] + ".yaml")
                input_templates["design"] = yaml.load(open(arch_yaml), Loader=yaml.SafeLoader)

                for key, cfg in dp["configurations"].items():
                    if key == "sparsity_schemes":
                        continue
                    cfg_yaml = os.path.join(template_root, "configurations", key, cfg + ".yaml")
                    input_templates[key] = (yaml.load(open(cfg_yaml), Loader=yaml.SafeLoader)
                                            if os.path.exists(cfg_yaml) else cfg)

                input_templates["sparsity_schemes"] = deepcopy(dp["processed_schemes"])

                hw_design_point_run(input_templates, out_dir)
                experiment_record[exp_key] = out_dir

if __name__ == '__main__':

    parser = argparse.ArgumentParser(""" sweep various workloads (DNN layers) for different accelerator designs """)
    parser.add_argument('--workload', type=str, required=True, choices=['resnet50_selected', 'alexnet_selected', 'mobilenetv2_selected'],
                        help='Name of the model workload to run (e.g., resnet50_selected).')
    parser.add_argument('-d', '--design_point', type=str, default= "MICRO22-STC-Case-Study", help="design point file to run, any file name in ../arch_templates/HW_systems/design_points/...")
    parser.add_argument('-o', '--output_dir', type=str, default=os.path.join(this_directory, os.pardir, "outputs"), help='top level output directory' )
    parser.add_argument('--layer_idx', type=int, default=None, help='specific layer idex to run')
    parser.add_argument('--density_degrees', nargs = "*", type=float, default=[1.0, 0.5, 0.3333, 0.25], help='weight density degrees to run')
    parser.add_argument('--max_layers', type=int, default=100, help='max number of layers to run')
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('--search_mapping', action="store_true", help="explore mappings instead of use the mappings already found")
    options = parser.parse_args()
    
    #vars setup from args
    MAX_LAYERS = options.max_layers
    OVERWRITE = True
    VERBOSE = options.verbose
    PRECISION = "int8"
    USE_MODEL = not options.search_mapping
    
    workload = options.workload 
    sweep_name = options.design_point        
    os.environ["SYSTEM_TEMPLATES_ROOT"] = os.path.abspath(os.path.join(this_directory, '../architecture_templates/HW_systems'))
    os.environ["RESULTS_ROOT"] = os.path.abspath(options.output_dir) # Use parsed output dir
    os.environ["gemm-ABZ"] = os.path.abspath("../workload/shapes/gemm-ABZ.yaml")
    os.environ["WORKLOADS"] = os.path.abspath("../workload/")
    
    print(f"=== Running Sweep for Model: {workload} ===")
    top(workload, options.density_degrees, VERBOSE, MAX_LAYERS, options.layer_idx)
