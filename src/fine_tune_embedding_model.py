from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import torch
from sentence_transformers.readers import InputExample

# Load model
model = SentenceTransformer("all-mpnet-base-v2")

# Load dataset
dataset = load_dataset("json", data_files="nutrition_data.json")
train_dataset = dataset["train"]

# Ensure the Relevance Score is float
train_examples = []
for i in range(len(train_dataset)):
    question = train_dataset['Question'][i]
    answer = train_dataset['Answer'][i]
    score = float(train_dataset['Relevance Score'][i])  # Convert to float explicitly
    train_examples.append(InputExample(texts=[question, answer], label=score))

# Loss function
loss = CosineSimilarityLoss(model=model)

# Training Arguments
args = SentenceTransformerTrainingArguments(
    output_dir="fine_tuned_embedding_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    save_total_limit=2,
    logging_steps=100,
    eval_steps=100,
    # fp16=True
)

# Evaluator - Ensure Relevance Score is float
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=train_dataset["Question"],
    sentences2=train_dataset["Answer"],
    scores=[float(score) for score in train_dataset["Relevance Score"]]  # Convert scores to float explicitly
)

# Create DataLoader for the train dataset
train_dataloader = DataLoader(train_examples, batch_size=16)

# Start training using the model's fit method
model.fit(
    train_objectives=[(train_dataloader, loss)],
    evaluator=evaluator,
    epochs=3,
    warmup_steps=100,
    output_path="fine_tuned_embedding_model"
)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_embedding_model/model")
