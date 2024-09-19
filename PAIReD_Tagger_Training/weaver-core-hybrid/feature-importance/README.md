# Evaluating the feature importance

This folder is dedicated to evaluate the feature importance of trained networks. In the following you can find a quick recipe on how to do the evaluation.

## Recipe
You have to install the `captum` package for python:
```bash
pip install captum
```
Then, run with something like this:
```bash
python evaluate_feature_importance.py path/to/model_state.pt \
--network-config path/to/network/config.py \
--data-config path/to/data/config.yaml \
--datafile path/to/sample/PAIReD/data/file/PAIReD_VHcc.root \
--Njets 100 \
--selection "(label_CC==1)&(MC_vector_flav!=15)" \
--target label_CC
```

**Example:**
```bash
python evaluate_feature_importance.py "../../trained-models/PAIReDEllipse 3 SV [DY]/model_state.pt" --network-config "../../networks/PAIReD_ParT_sv_classifier.py" --data-config ../../dataconfigs/PAIReD_classifier_sv.train.yaml --datafile ../../../PAIReD_Data_Production/PFNano_to_PAIReD/data/example_PAIReD_mcRun3_EE.root
```