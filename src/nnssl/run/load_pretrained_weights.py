import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            if key not in pretrained_dict:
                if verbose:
                    print(f"Key {key} is missing in the pretrained model weights. Skipping...")
                continue

            if model_dict[key].shape != pretrained_dict[key].shape:
                if (".stem.convs.0.conv.weight" in key or ".stem.convs.0.all_modules.0.weight" in key) and \
                   model_dict[key].shape[1] == 4 and pretrained_dict[key].shape[1] == 1:
                    if verbose:
                       print(f"Adapting {key} from 1 to 4 channels by repeating weights.")
                    model_dict[key].copy_(pretrained_dict[key].repeat(1, 4, 1, 1, 1))
                else:
                    if verbose:
                        print(f"Shape mismatch for key {key}: pretrained {pretrained_dict[key].shape}, "
                              f"model {model_dict[key].shape}. Skipping...")
                continue

            model_dict[key].copy_(pretrained_dict[key])

    mod.load_state_dict(model_dict)
    print("################### Loading pretrained weights from file ", fname, '###################')
    return mod


