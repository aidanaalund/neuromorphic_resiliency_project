import numpy as np
import torch.nn as nn
import snntorch as snn
import yaml
from collections import OrderedDict
import os

# Author: Aidan Aalund
# Code for the yaml dictionary and dumping of the dvs example

def init_dictionary(name = 'output'):
    return OrderedDict({
        'network': OrderedDict({
            'name': name,
            'groups': [],
            'edges': []
        }),
        'mappings': []
    })

class OrderedDictDumper(yaml.SafeDumper):
    pass

def ordered_dict_representer(dumper, data):
    return dumper.represent_dict(data.items())

def ndarray_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data.tolist(), flow_style=True)

def dump_yaml(dictionary, name='output'):
    OrderedDictDumper.add_representer(OrderedDict, ordered_dict_representer)
    OrderedDictDumper.add_representer(np.ndarray, ndarray_representer)
    
    # Get the directory of the current file (utils.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'saved_yamls')
    
    # print(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, name + '.yaml')
    print(f"Output file path: {output_file}")
    try:
        with open(output_file, 'w') as file:
            yaml.dump(dictionary, file, Dumper=OrderedDictDumper, sort_keys=False, default_flow_style=False, indent=2)
        print(f"Done! File saved to {output_file}")
    except Exception as e:
        print(f"Error writing file: {e}")

# Code for adding neurons groups and edges
def add_neuron(dictionary, group_name, num_neurons, model_attributes, neuron_attributes):
    # TODO: neuron_dict needs to be edited either by hand or manually for spikes, poisson, rate input encoding
    neuron_dict = [{'0..'+str(num_neurons-1): neuron_attributes} if num_neurons > 1 else {'0': neuron_attributes}]
    dictionary['network']['groups'].append(OrderedDict({
        'name': group_name,
        'attributes': model_attributes,
        'neurons': neuron_dict
    }))

def initialize_group(dictionary, group_name, attributes):
    dictionary['network']['groups'].append(OrderedDict({
        'name': group_name,
        'attributes': attributes,
        'neurons': []
    }))

def add_individual_neuron(dictionary, group_number, neuron_id, attributes):
    dictionary['network']['groups'][group_number]['neurons'].append(OrderedDict({
        neuron_id: attributes
    }))

# prev_edges is a 2D numpy array where prev_edges[i][j] 
# represents the weight of the connection from neuron j 
# in the previous layer to neuron i in the current layer.
def add_edges(dictionary, num_layers, num_neurons, prev_edges, input_found = True):
    s = f'group{num_layers-1}'
    
    for i in range(len(prev_edges)):
        for j in range(len(prev_edges[i])):

            weight = prev_edges[i][j] # shape of prev edges is out_features, in_features

            dictionary['network']['edges'].append(OrderedDict({
                s+f'.{j} -> group{num_layers}.{i}': [{
                    'weight': float(weight)
                }]
            }))

def add_edges_conv(dictionary, num_layers, attributes, input_found = True):
    s = f'group{num_layers-1}'
    
    dictionary['network']['edges'].append(OrderedDict({
                s+f' -> group{num_layers}': [attributes]
            }))

def add_edges_maxpool2d(dictionary, num_layers, attributes, input_found = True):
    s = f'group{num_layers-1}'
    
    dictionary['network']['edges'].append(OrderedDict({
                s+f' -> group{num_layers}': [attributes]
            }))

# TODO: refactor to maybe have more efficient mappings??? Wrote some code to track total count
def add_hw_mapping(dictionary, num_neurons, num_layers):
    # if not hasattr(add_hw_mapping, "total_neurons"):
    #     add_hw_mapping.total_neurons = 0
    # add_hw_mapping.total_neurons += num_neurons
    # add_hw_mapping.total_neurons %= 1023

    dictionary['mappings'].append(OrderedDict({
                f'group{num_layers}.0..{num_neurons-1}': {'core': float(f'{num_layers}.0')} # TODO: this float cast is weird
            }))

# TODO: cannot handle MaxPool2d, MaxPool1d, and sort of Conv2d at the moment.
def identify_layer_type(dictionary, layer, input_found, num_previous_layer_outputs, num_layers, prev_edges, input = [{'spikes': [1,0,0,0]}], next_layer = None, prev_layer = None):
        if isinstance(layer, nn.Linear):
            print("Linear")
            if not input_found: # Case of first layer
                num_previous_layer_outputs = layer.in_features
                prev_edges = layer.weight.detach().numpy()
                model_attributes = [{'soma_hw_name' : 'loihi_inputs'}, {'log_spikes': True}]
                neuron_attributes = input #TODO: This will need to be reworked
                add_neuron(dictionary, f'group{num_layers-1}', num_previous_layer_outputs, model_attributes, neuron_attributes)
                # TODO: commented code adds ability to add individual neurons
                #initialize_group(dictionary, f'group_{num_layers}', model_attributes)
                #for i in range(num_previous_layer_outputs):
                #    add_individual_neuron(dictionary, num_layers-1, i, [{'spikes' : [1,0,0,0]}])
                add_hw_mapping(dictionary, num_previous_layer_outputs, num_layers-1)
                input_found = True
            # Get the number of output features (neurons) of the current linear layer
            num_previous_layer_outputs = layer.weight.shape[0] 
            prev_edges = layer.weight.detach().numpy()
        elif isinstance(layer, snn.Leaky):
            print("Leaky")
            num_neurons = num_previous_layer_outputs
            attributes = []
            if(isinstance(prev_layer, nn.MaxPool2d)):
                attributes = [{
                'threshold': float(layer.threshold),
                'compartments': prev_layer.kernel_size*prev_layer.kernel_size,},
                {'compartment_in_ops':["skip", "pop", "pop", "pop"]}, #- compartments: 4 # c1, c2, c3, c4
                {'compartment_join_ops': ["skip", "max", "max", "max"]},
                {'compartment_out_ops': ["push", "push", "push", "skip"]}]
            else:
                attributes = [{
                'threshold': float(layer.threshold),
                }]
            
            add_neuron(dictionary, f'group{num_layers}', num_neurons, attributes, [])
            add_hw_mapping(dictionary, num_neurons, num_layers)
            if not input_found:
                input_found = True
            else:
                if prev_edges.ndim == 2: # TODO: hacky way of preventing crashes... also doesn't work.
                    add_edges(dictionary, num_layers, num_neurons, prev_edges)
            num_layers += 1
        elif isinstance(layer, nn.Flatten):
            print("Flatten") # TODO: if dimension is higher than 1, flatten it. Also handle input_found = F/T cases?
        elif isinstance(layer, nn.Conv2d):
            print("Conv2D")
            # TODO: Calculate total neurons: output_channels * output_height * output_width
            # output_height = 32  # You'll need to track this through the network
            # output_width = 32   # You'll need to track this through the network
            # if prev_layer and isinstance(prev_layer, nn.MaxPool2d):
            #     output_height //= prev_layer.kernel_size
            #     output_width //= prev_layer.kernel_size
                
            # num_previous_layer_outputs = layer.out_channels * output_height * output_width
            num_previous_layer_outputs = layer.weight.shape[0] # number of output channels (recall N,Cin,Hin,Win)
            prev_edges = layer.weight.detach().numpy()
            weights_transposed = np.transpose(prev_edges, (0, 2, 3, 1)) # dimensions are NHWC
            weights_flattened = weights_transposed.flatten() # dimensions are N+H+W+C
            attributes = {
                'type': 'conv2d',
                'input_height': layer.weight.shape[2],
                'input_width': layer.weight.shape[3],
                'input_channels': layer.in_channels,
                'kernel_width' : layer.kernel_size[0],
                'kernel_height' : layer.kernel_size[1],
                'kernel_count': layer.out_channels,
                'stride_width': layer.stride[0],
                'stride_height': layer.stride[0],
                'weight': weights_flattened,
                # new stuff
                'maxpool2d': isinstance(next_layer, nn.MaxPool2d),
                'pool_width': next_layer.kernel_size, # This assumes a square shape
                'pool_height': next_layer.kernel_size,
                'pool_params': { 'compartment': [0,1,2,3] } # how to get this programatically?
            }
            if not input_found: # I don't think this case is possible with current workloads
                add_neuron(dictionary, 'group0', layer.in_channels, {'soma_hw_name' : 'loihi_inputs'}, [])
                input_found = True
            add_edges_conv(dictionary, num_layers, attributes, input_found)
        elif isinstance(layer, nn.MaxPool2d):
            print("MaxPool2D")
            #add_neuron(dictionary, f'group{num_layers}', num_neurons, attributes, [])
        elif isinstance(layer, nn.Dropout):
            print("Dropout") # Ignore dropout layers? during conversion, they are a function of trainings
        else:
            print("Unknown layer type")
        return num_previous_layer_outputs, prev_edges, input_found, num_layers

def convert_class_snn_to_yaml(net, name = 'output', input = [{'spikes': [1,0,0,0]}]): 
    input_found = False
    num_previous_layer_outputs = -1
    num_layers = 1
    prev_edges = []
    dictionary = init_dictionary(name)
    layers = list(net.named_children())

    for i, (_, layer) in enumerate(layers):
        # Get next layer if it exists
        next_layer = layers[i + 1][1] if i < len(layers) - 1 else None
        prev_layer = layers[i - 1][1] if i > 0 else None

        num_previous_layer_outputs, prev_edges, input_found, num_layers = identify_layer_type(dictionary, layer,input_found,
                                                                                              num_previous_layer_outputs,
                                                                                              num_layers,prev_edges,input,next_layer,prev_layer)
    dump_yaml(dictionary, name)


# if __name__ == "__main__":
