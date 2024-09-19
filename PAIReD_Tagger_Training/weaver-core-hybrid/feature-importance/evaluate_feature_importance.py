import torch
from captum.attr import IntegratedGradients
import json
import numpy as np
import awkward as ak
import uproot
from weaver.utils.data.preprocess import _build_new_variables
from weaver.utils.data.tools import _eval_expr
from weaver.utils.data.config import DataConfig
from weaver.utils.dataset import _finalize_inputs
from tools.load_model import load_model
import argparse



# parameters
parser = argparse.ArgumentParser(description="Evaluate the feature importance for a given network.")
parser.add_argument("network", type=str, help="Path to trained model (PyTorch network file)")
parser.add_argument('-n', '--network-config', type=str,
                    help='network architecture configuration file; the path must be relative to the current dir')
parser.add_argument('-o', '--network-option', nargs=2, action='append', default=[],
                    help='options to pass to the model class constructor, e.g., `--network-option use_counts False`')
parser.add_argument('-d', '--datafile', type=str,
                    help='validation data file')
parser.add_argument('-c', '--data-config', type=str,
                    help='data config YAML file')
parser.add_argument("-j", "--Njets", type=int, default=100,  help="Number of jets to test the network on (default is 100)")
parser.add_argument("-s", "--selection", type=str, default="(label_CC==1)&(MC_vector_flav!=15)",  help="Selection for jets to be used (default is '(label_CC==1)&(MC_vector_flav!=15)')")
parser.add_argument("-t", "--target", type=str, default="label_CC",  help="Selection for jets to be used (default is label_CC)")
args = parser.parse_args()


print(50*"*")
print(" Start evaluation of feature importance")
print(50*"*", "\n")
print("network:", args.network)
print("data config:", args.data_config)
print("number of jets:", args.Njets)
print("selection cuts:", args.selection)
print("target label:", args.target)


###############################
# Load the data configuration
###############################
print("\nLoad data configuration...")
data_config = DataConfig.load(args.data_config, load_observers=False)


###########################
# Load the trained model  
###########################
print("Load network...")
model = load_model(args, data_config)


###########################
# Load ROOT data files
###########################
print("Load and preprocess data...")
with uproot.open(args.datafile + ":tree") as tree:
    table = tree.arrays(["MC_higgs_pt", "MC_vector_flav", "MC_higgs_mass", "label_CC"])
    selected = ak.values_astype(_eval_expr(args.selection, table), 'bool')
    i = np.argmax((np.cumsum(selected) >= args.Njets) *1)
    # load exactly the wanted number of selected PAIReD jets
    table = tree.arrays(cut=args.selection, entry_stop=i+1)

# add any newly defined variables from the config file
table = _build_new_variables(table, {k: v for k, v in data_config.var_funcs.items()})

# preprocess input from ROOT files
inputs = _finalize_inputs(table, data_config)
inputs_ = [torch.from_numpy(inputs["_%s" % input_name]).to(torch.float32) for input_name in data_config.input_names]


###########################
# Run prediction
###########################
print("\nRun prediction as cross check...")
output = model(*inputs_)
output = torch.nn.functional.softmax(output, dim=1)
print("done...")
print("Output prob of", data_config.options['labels']['value'], "(first 5 PAIReD jets):")
print(output[:5])


##############################
# Run feature importance eval
##############################
# Number of jets
N = len(table)
print("\nRun Integrated Gradients on", N, args.target, "jets in total...")
# set target index
target = data_config.options['labels']['value'].index(args.target)

# Initialize the attribution algorithm with the model
integrated_gradients = IntegratedGradients(model)

# set array for feature importance
importance = {input_name : torch.zeros(len(inputs["_%s" % input_name][0])) for input_name in data_config.input_names}

for i in range(N):
    # Ask the algorithm to attribute our output target to
    input_i = tuple([input_[i:(i+1)] for input_ in inputs_])
    attributions_ig = integrated_gradients.attribute(input_i, target=target, n_steps=100)
    for j, key in enumerate(importance.keys()):
        print(key)
        importance[key] += torch.sum(attributions_ig[j], (2,0)).abs()

# save importance scores with feature name in json file
importance_dict = {}
for key in importance.keys():
    for i, feature in enumerate(data_config.input_dicts[key]):
        importance_dict[feature] = float(importance[key][i] / N)  # divide by the number of PAIReD jets to get the average
outputpath = args.network[:-3] + "_feature_importance.json"
print("\nsave feature importance scores in here:", outputpath)
with open(outputpath, 'w') as f:
    json.dump(importance_dict, f)
