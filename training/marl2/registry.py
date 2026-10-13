from environment.marl_adapter import JackalMultiAgentEnv
from training.marl2.learners.qmix_learner import QMixLearner


ENV_REGISTRY = {
    "jackal": JackalMultiAgentEnv,
}

LEARNER_REGISTRY = {
    "qmix": QMixLearner,
}
