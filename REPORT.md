# Lakeshore Ticket Sorter: findings

## Result

At the largest size I tried (600 training messages), full fine-tuning
of DistilBERT reached a mean validation accuracy of 0.911 (range 0.904
to 0.921 over three seeds), ahead of the TF-IDF baseline at 0.858
(range 0.850 to 0.871). On the held-out test set, the fine-tuned model
scored 0.900 accuracy (macro F1 0.903) against 0.846 for TF-IDF and
0.292 for always predicting the most common queue.

## Where fine-tuning helped

The learning curve shows the gap depends on how much labelled data
there is. At 100 messages, TF-IDF averaged 0.761 and full fine-tuning
0.739, and their ranges overlapped, so I cannot say fine-tuning was
better there. At 300 messages, full fine-tuning moved ahead (0.896
against 0.863), though the ranges still overlapped a little. Only at
600 messages did full fine-tuning clearly beat TF-IDF, with ranges that
did not overlap. Head-only fine-tuning, where only the new
classification layer trains, was the weakest of the three at 300 and
600 messages (about 0.80), below TF-IDF both times.

## Why

Two reasons fit the evidence. First, DistilBERT is a large model, and
100 messages are likely too few for it to adapt, which is why TF-IDF, a
simple word-counting method, matched it at that size. I checked this
against the learning curve: as the training size grew, full fine-tuning
kept improving while TF-IDF levelled off. Second, the generated
messages repeat a small set of sentence patterns, which suits TF-IDF. I
checked this with my own 40 messages below. Seed noise also matters at
small sizes: at 100 messages the full fine-tuning range was wide (0.700
to 0.775), so small gaps there are not reliable. The learning rates
were chosen on validation rows (5e-05 for full, 0.0005 for head-only),
so the head-only gap is not just a poor rate.

## My own 40 messages

I scored the final model once on 40 messages I wrote myself, ten per
queue, in different wording. It reached 0.900 accuracy, the same as on
the test set. That suggests the generated data does not flatter the
model much, at least on my wording. Forty messages is a small check, so
I would not treat this as proof.

## Recommendation

If you can label around 600 messages, full fine-tuning is worth it: it
beat TF-IDF by about five points, and the seed ranges did not overlap.
If you only have about 100 labelled messages, use TF-IDF, which was
just as accurate and is far simpler and cheaper to run. One thing that
would change this: if real customer messages use very different wording
from the invented ones, the fine-tuned model's advantage could shrink,
and TF-IDF might be the safer choice.

## Honest limit

All the training, validation, and test messages were invented from
templates, not collected from real customers, so I cannot tell you how
either method will do on real messages. About 12 percent of the
messages are vague and cannot be sorted from their text at all, which
puts a ceiling on accuracy that I cannot measure exactly. With only
three seeds, the ranges are a rough guide, not a precise one.