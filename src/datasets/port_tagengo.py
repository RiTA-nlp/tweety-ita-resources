import fire
from datasets import load_dataset


def format_batch(batch):
    messages = list()
    for conv in batch["conversations"]:
        chat = [
            {
                "role": "user" if m["from"] == "human" else "assistant",
                "content": m["value"],
            }
            for m in conv
        ]
        messages.append(chat)
    batch["messages"] = messages
    return batch


def main():
    data = load_dataset(
        "lightblue/tagengo-gpt4",
        split="train",
        trust_remote_code=True,
        num_proc=4,
    )
    # keep only italian documents
    data = data.filter(lambda x: x["language"] == "Italian", num_proc=4)

    data = data.map(
        format_batch, batched=True, num_proc=4, remove_columns=["conversations"]
    )

    data.push_to_hub("RiTA-nlp/tagengo-gpt4-italian")


if __name__ == "__main__":
    fire.Fire(main)
