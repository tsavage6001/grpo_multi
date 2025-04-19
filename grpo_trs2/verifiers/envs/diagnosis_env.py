from datasets import load_dataset
from verifiers.rewards.reward_function import reward_function
from verifiers.envs.environment import Environment

class DiagnosisEnv(Environment):
    def get_dataset(self, **kwargs):
        return load_dataset("csv", data_files="patient_dataset.csv")["train"]

    def get_eval_dataset(self, **kwargs):
        return None

    def get_reward_funcs(self, **kwargs):
        return [reward_function]

    def get_reward_weights(self, **kwargs):
        return [1.0]
