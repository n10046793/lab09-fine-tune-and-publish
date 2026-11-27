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
