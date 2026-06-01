from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import MODELS
from cache import setup_cache
from accuracy import evaluate_model_on_dataset_lama, evaluate_model_on_dataset_trivia_qa

if __name__ == "__main__":
    cache_dir = setup_cache()

    for model_id in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )
        accuracy = evaluate_model_on_dataset_lama(tokenizer, model, max_examples=500)

        print(f"Model: {model_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print("---")

        accuracy = evaluate_model_on_dataset_trivia_qa(tokenizer, model, max_examples=500)

        print(f"Model: {model_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print("---")
