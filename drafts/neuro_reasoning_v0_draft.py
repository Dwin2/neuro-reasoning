import verifiers as vf
import random


def load_environment(**kwargs) -> vf.Environment:
    """
    Load this environment.

    Environments typically return vf.SingleTurnEnv, vf.ToolEnv, etc.
    """
    raise NotImplementedError("Implement load_environment here.")

def sample_params():
    rate = [256, 512, 1000]
    sampling_rate = random.choice(rate)
    cutoff = random.randint(10, sampling_rate)

    return {
        "rate": sampling_rate,
        "cutoff": cutoff
    }

def make_question(params):
    return f"An EEG system samples at {params['rate']} Hz with a cutoff frequency of {params['cutoff']} Hz. Is this a valid cutoff? Answer yes or no."

def verify(params, answer):
    if answer.lower() == "yes":
        return params['cutoff'] < params['rate'] / 2
    elif answer.lower() == "no":
        return params['cutoff'] >= params['rate'] / 2
    else:
        raise ValueError("Answer must be 'yes' or 'no'.")