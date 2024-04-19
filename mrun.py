import argparse
import json
import os
from smiles_rl.configuration_envelope import ConfigurationEnvelope


from typing import Optional, Dict, Type, Any

from smiles_rl.model.actor_model import ActorModel
from smiles_rl.model.actor_model_transformer import ActorModelTransformer
from smiles_rl.utils.general import set_default_device_cuda


import importlib

from smiles_rl.agent.base_agent import BaseAgent
from distill import *

from dacite import from_dict

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def load_dynamic_class(
    name_spec: str,
    default_module: Optional[str] = None,
    exception_cls: Type[Exception] = ValueError,
):

    if name_spec is None:
        raise KeyError(f"Key method not found in scoring function config")

    if "." not in name_spec:
        name = name_spec
        if not default_module:
            raise exception_cls(
                "Must provide default_module argument if not given in name_spec"
            )
        module_name = default_module
    else:
        module_name, name = name_spec.rsplit(".", maxsplit=1)
    try:
        loaded_module = importlib.import_module(module_name)
    except ImportError:
        raise exception_cls(f"Unable to load module: {module_name}")

    if not hasattr(loaded_module, name):
        raise exception_cls(
            f"Module ({module_name}) does not have a class called {name}"
        )

    return getattr(loaded_module, name)


def run(
    config: ConfigurationEnvelope,
    agent,
) -> None:
    print("Starting run", flush=True)

    batch_size = config.reinforcement_learning.parameters.batch_size
    n_steps = config.reinforcement_learning.parameters.n_steps
    for _ in range(n_steps):
        _run(batch_size, agent)
    torch.save(agent._actor.transformer.state_dict(), 'post_rl_transformer_weights.pth')
    agent.log_out()


def _run(
    batch_size: int,
    agent: BaseAgent,
) -> None:

    smiles = agent.act(batch_size)

    assert len(smiles) <= batch_size, "Generated more SMILES strings than requested"

    agent.update(smiles)


def _read_json_file(path: str) -> Dict[str, Any]:
    """Reads json config file

    Args:
        path (str): Path to json file with configuration

    Returns:
        dict: Dictionary containing configurations from json file
    """

    print("Rading JSON file", flush=True)
    with open(path) as f:
        json_input = f.read().replace("\r", "").replace("\n", "")
    try:
        config = json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")

    return config


def _construct_logger(config: ConfigurationEnvelope):
    """Creates logger instance

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        logger instance
    """

    name_spec = config.logging.method

    if name_spec is not None:
        method_class = load_dynamic_class(name_spec)
    else:
        raise KeyError(f"Key method not found in logging config")

    logger = method_class(config)

    return logger


def _construct_scoring_function(config: ConfigurationEnvelope):
    """Creates scoring function instance

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        scoring function instance
    """

    name_spec = config.scoring_function.method

    method_class = load_dynamic_class(name_spec)

    scoring_function = method_class(config)

    return scoring_function

def _knowledge_transfer(config: ConfigurationEnvelope):
    teacher_model_path = config.reinforcement_learning.parameters.agent
    teacher_model = ActorModel.load_from_file(file_path=teacher_model_path, sampling_mode=False)
    student_model = ActorModelTransformer.load_from_file(pre_training_file_path=teacher_model_path, transfer_weight_path='transformed_model_weights_6000.pth')

    learning_rate = 1e-5
    weight_decay = 1e-5
    kt_optimizer = optim.Adam([
        {'params': [p for n, p in student_model.transformer.decoder.named_parameters() if 'embed' not in n], 'weight_decay': weight_decay},
        {'params': student_model.transformer.out.parameters(), 'weight_decay': weight_decay},
        {'params': [p for n, p in student_model.transformer.decoder.named_parameters() if 'embed' in n], 'weight_decay': 0}
    ], lr=learning_rate)
    distiller = ModelDistillation(kt_optimizer, alpha=0.5, batch_size=200, n_steps=2000, temp=0.99) 

    distiller.transfer(student_model, teacher_model)  

    transformed_weights_path = 'transformed_model_weights.pth'
    torch.save(student_model.state_dict(), transformed_weights_path)
    KT_loss = f'KT_loss_{learning_rate}_{weight_decay}_{distiller.batch_size}_{distiller.temp}_{distiller.n_steps}'
    distiller.save_losses_to_csv(filename=KT_loss)
    return transformed_weights_path

def _construct_agent(config: ConfigurationEnvelope, logger, scoring_function, diversity_filter, replay_buffer) -> BaseAgent:
    name_spec = config.reinforcement_learning.method
    method_class = load_dynamic_class(name_spec)
    transformed_weights_path = _knowledge_transfer(config)
    transformed_weights_path = 'transformed_model_weights.pth'
    agent = method_class(config, scoring_function, diversity_filter, replay_buffer, logger)
    if os.path.exists(transformed_weights_path):
        model = agent._actor.transformer
        model.load_state_dict(torch.load(transformed_weights_path))
        model.freeze_layers(['embed'])
        torch.save(model.state_dict(), 'pre_rl_transformer_weights.pth')
        print("Transformer transfer knowledge complete.")
    return agent


def _construct_diversity_filter(config: ConfigurationEnvelope):
    """Creates diversity filter instance

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        diversity filter instance
    """
    name_spec = config.diversity_filter.method

    method_class = load_dynamic_class(name_spec)

    diversity_filter = method_class(config)

    return diversity_filter


def _construct_replay_buffer(config: ConfigurationEnvelope):
    """Create replay buffer instance

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        replay buffer instance
    """
    name_spec = config.replay_buffer.method

    method_class = load_dynamic_class(name_spec)

    replay_buffer = method_class(config.replay_buffer.parameters)

    return replay_buffer


def _construct_run(config: ConfigurationEnvelope) -> BaseAgent:
    """Construct run and returns agent

    Args:
        config (ConfigurationEnvelope): configuration settings

    Returns:
        BaseAgent: agent
    """

    # Set default device of pytorch tensors to cuda
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

    logger = _construct_logger(config)
    scoring_function = _construct_scoring_function(config)

    diversity_filter = _construct_diversity_filter(config)

    replay_buffer = _construct_replay_buffer(config)

    agent = _construct_agent(
        config, logger, scoring_function, diversity_filter, replay_buffer
    )

    return agent


def _get_arguments() -> argparse.Namespace:
    """Reads command-line arguments

    Returns:
        argparse.Namespace: command-line arguments
    """

    print("Getting input args", flush=True)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="path to config json file",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    print("Starting main", flush=True)

    # Get command line arguments
    args = _get_arguments()

    # Read configuration file
    config_json = _read_json_file(args.config)

    # Create envelope of configuration
    config = from_dict(data_class=ConfigurationEnvelope, data=config_json)

    # Construct run
    agent = _construct_run(config)

    run(config, agent)


if __name__ == "__main__":
    main()
