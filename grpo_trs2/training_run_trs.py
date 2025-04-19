from transformers import AutoTokenizer
from verifiers.envs.diagnosis_env import DiagnosisEnv
from verifiers.inference.vllm_client import VLLMClient
from trl import GRPOConfig
from trainer.grpo_env_trainer import GRPOEnvTrainer

# Init tokenizer and environment
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.1")
env = DiagnosisEnv(tokenizer=tokenizer)

# Load datasets and reward function
train_dataset = env.get_dataset()
reward_funcs = env.get_reward_funcs()
reward_weights = env.get_reward_weights()

# Init config
grpo_args = GRPOConfig(
    output_dir="./grpo_finetuned",
    per_device_train_batch_size=2,
    num_generations=4,
    num_iterations=2,
    max_prompt_length=512,
    max_completion_length=64,
    beta=0.1,
    use_vllm=True,
    logging_steps=1,
)

# Init vLLM-backed models
assistant_llm = VLLMClient("TinyLlama/TinyLlama-1.1B-Chat-v0.1")
frozen_llm = VLLMClient("eljanmahammadli/micro-llama-300M-v1")

trainer = GRPOEnvTrainer(
    model="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
    env=env,
    reward_funcs=reward_funcs,
    train_dataset=train_dataset,
    args=grpo_args
)

trainer.env.tokenizer = tokenizer
trainer.train()
