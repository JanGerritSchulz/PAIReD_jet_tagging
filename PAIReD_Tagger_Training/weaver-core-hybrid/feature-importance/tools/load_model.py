import torch
import ast
from importlib.util import spec_from_file_location, module_from_spec


def import_module(path, name='_mod'):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



def load_model(args, data_config):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config, name='_network_module')
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    model, model_info = network_module.get_model(data_config, **network_options)
    model_state_dict = torch.load(args.network, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    model.eval()
    return model