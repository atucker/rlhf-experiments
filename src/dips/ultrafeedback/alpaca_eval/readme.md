To run alpaca eval validation:
1. export OPENAI_API_KEY=<your_api_key>
2. Run blocks within `alpaca_eval.ipynb` to download the LoRA weights and generate responses.
3. Save outputs to named json files.
4. `alpaca_eval --model_outputs <path_to_json_file> --reference_outputs <path_to_json_file>` to run the evaluation.