# Lakeshore Ticket Sorter: fine-tuning against a baseline

Model: https://huggingface.co/n10046793/lab09-bike-share-ticket-router

## What it does

This project sorts Lakeshore Bike Share support messages into four
queues: billing, bike_fault, station, and account. It compares a
fine-tuned DistilBERT model against two simple baselines, on the same
training rows and the same validation rows, to answer one question: does
fine-tuning help on this data, and by how much?

## How to run it

Create the environment and activate it:

```
conda env create -f environment.yml
conda activate lab09-fine-tune-and-publish
```

Run the experiment grid (writes reports/results.csv and
reports/learning_curve.png):

```
python run_experiments.py
```

Train the final model and score the test rows and the 40 own messages
(writes reports/final.json and reports/confusion_matrix.png, and saves
the model to model/):

```
python final_check.py
```

Write the model card and publish the model (publishing needs a Hugging
Face login with `hf auth login` first):

```
python share.py
python publish.py
```

Run the checks:

```
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

The model/ folder is not committed. Recreate it by running
final_check.py, which trains the chosen model and saves it there.

## Data and split

The data comes from tickets_data.py, which generates 1,200 invented
messages with a label. About 12 percent are vague, about 10 percent mix
two topics, and about 8 percent contain a typing slip, so no model can
reach 100 percent. I made one stratified 60/20/20 split with a fixed
seed: 720 training rows, 240 validation rows, and 240 test rows. The
test rows were used once, at the end. I also wrote 40 messages myself
(data/own_messages.csv), ten per queue, used only once at the end.

## Methods

Two baselines: always predicting the most common training label, and
TF-IDF with logistic regression. The fine-tuned model is
distilbert/distilbert-base-uncased with a new four-label head, trained
with the Hugging Face Trainer. I compared two variants: head-only (the
DistilBERT body frozen, only the new head trains) and full (the whole
model trains). Learning rates were chosen on the validation rows: 5e-05
for full and 0.0005 for head-only, each for 3 epochs in the final run.
Every method was trained on the same sampled rows and scored on the
same validation rows, over three seeds and three training sizes (100,
300, and 600).

## Results

Mean validation accuracy over three seeds (range in brackets):

| Method | 100 | 300 | 600 |
|---|---|---|---|
| most_common | 0.288 | 0.288 | 0.288 |
| tfidf | 0.761 (0.750-0.771) | 0.863 (0.821-0.900) | 0.858 (0.850-0.871) |
| head | 0.729 (0.721-0.733) | 0.806 (0.800-0.813) | 0.804 (0.792-0.817) |
| full | 0.739 (0.700-0.775) | 0.896 (0.871-0.908) | 0.911 (0.904-0.921) |

The learning curve is in reports/learning_curve.png. Full fine-tuning
only clearly beat TF-IDF at 600 messages, where the seed ranges did not
overlap.

Final scores on the held-out test set (240 messages), with the chosen
model (full fine-tuning) trained on all 720 training rows:

| Method | Accuracy | Macro F1 |
|---|---|---|
| full | 0.900 | 0.903 |
| tfidf | 0.846 | 0.846 |
| most_common | 0.292 | 0.113 |

The confusion matrix is in reports/confusion_matrix.png.

## Your own 40 messages

The final model scored 0.900 accuracy on my 40 hand-written messages,
the same as on the test set. That suggests the generated data does not
flatter the model much, though 40 messages is a small check.

## Publishing

The fine-tuned model and its model card are published at the link at the
top of this file. The card lists the intended use, training data,
results, and limitations, and states that the training text was
invented. The Gradio app and Space (step 10 of the brief) are not
required for my group, so they are not included.

## Tests

The tests/ folder has 13 test functions (15 cases with parametrization),
covering the split sizes and overlap, the small-sample function, the
baselines, tokenization, freezing the body, full fine-tuning changing
the body, seed repeatability, memorizing a small batch, saving and
reloading the model, and the model card headings. Run them with
`python -m pytest`.

## Limitations

The training text was invented from templates, not collected from real
customers, so results may not transfer to real messages. About 12
percent of messages are vague and cannot be sorted from their text,
which caps accuracy. Only three seeds were used, so the ranges are
rough. AI assistance was used to help write the code, tests, and these
documents; I ran, checked, and reviewed all of it, and the 40 own
messages were drafted with AI help as the instructor permitted.
