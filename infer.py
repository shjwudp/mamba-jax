import argparse

from mamba_ssm_jax.modules.mamba_simple import FlaxMambaLMHeadModel
from transformers import AutoTokenizer

SHAKESPEARE_PROMPT = """ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    model = FlaxMambaLMHeadModel.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)


    # Encode the prompt and generate the output
    inputs = tokenizer.encode(SHAKESPEARE_PROMPT, return_tensors="jax")
    outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=0.7, num_return_sequences=3)

    # Decode and print the output
    for i, output in enumerate(outputs.sequences):
        print(f"Generated {i}: {tokenizer.decode(output, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
