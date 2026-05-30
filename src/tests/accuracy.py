import re
import unicodedata

def _normalize_text(text: str) -> str:
    normalized_text = unicodedata.normalize("NFKC", text)
    normalized_text = normalized_text.casefold()
    normalized_text = re.sub(r"[^a-z0-9\s]", " ", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    return normalized_text

def _evaluate_model_on_dataset(tokenizer, model, dataset_entries):
    counter = 0
    dataset_length = len(dataset_entries)

    for entry in dataset_entries:
        masked_sentence = entry.get("masked_sentences", "")[0]
        gold_answer = entry.get("obj_label", "")
        prompt = masked_sentence.split("[MASK]")[0]

        tokenized_inputs = tokenizer(prompt, return_tensors="pt")
        tokenized_inputs = {key: value.to(model.device) for key, value in tokenized_inputs.items()}

        generated_ids = model.generate(
            **tokenized_inputs,
            max_new_tokens=15,
            pad_token_id=tokenizer.eos_token_id,
        )

        input_length = tokenized_inputs["input_ids"].shape[-1]
        predicted_ids = generated_ids[0][input_length:]
        predicted_text = tokenizer.decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        is_correct = _normalize_text(predicted_text).__contains__(_normalize_text(gold_answer))
        if is_correct:
            counter += 1
            # print("------------------")
            # print(gold_answer)
            # print(predicted_text)
            # print("------------------")

    accuracy = counter / dataset_length if dataset_length else 0.0
    return counter, dataset_length, accuracy