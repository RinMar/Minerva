"""
Triplet extraction using Rebel.
Used to parse unstructured text into structured subject-relation-object triplets
for the knowledge graph.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.paths import MODELS_DIR
from src.config import config

model_name = config["models"].get("triplet_model_id", "Babelscape/rebel-large")

device = config["llm"].get("device", "cpu")
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=MODELS_DIR).to(device)
model.eval()


def extract_triplets(text):
    # tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    # generate
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            do_sample=False
        )

    # decode
    extracted_text = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=False
    )

    # parse triplets
    triplets = []
    subject, relation, object_ = "", "", ""
    current = None

    tokens = extracted_text.replace("<s>", "").replace("</s>", "").strip().split()

    for token in tokens:
        if token == "<triplet>":
            if subject and relation and object_:
                triplets.append({
                    "head": subject.strip(),
                    "type": relation.strip(),
                    "tail": object_.strip()
                })
            subject, relation, object_ = "", "", ""
            current = "subject"

        elif token == "<subj>":
            current = "object"

        elif token == "<obj>":
            current = "relation"

        else:
            if current == "subject":
                subject += " " + token
            elif current == "object":
                object_ += " " + token
            elif current == "relation":
                relation += " " + token

    # final triplet
    if subject and relation and object_:
        triplets.append({
            "head": subject.strip(),
            "type": relation.strip(),
            "tail": object_.strip()
        })

    return triplets
