"""Run the experiment grid and write the reports."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

from baselines import most_common_label, tfidf_model
from finetune import train_model
from prepare import as_lists, make_split, small_sample

SIZES = [100, 300, 600]
SEEDS = [1, 2, 3]
EPOCHS = {"head": 5, "full": 3}
LR_CANDIDATES = {"head": [5e-4, 1e-3], "full": [2e-5, 5e-5]}


def best_epoch(history):
    """The epoch with the best validation accuracy."""
    return max(history, key=lambda row: row["val_accuracy"])


def choose_learning_rates(train_df, val_texts, val_labels):
    """Pick a learning rate for each variant using validation accuracy."""
    sample = small_sample(train_df, 300, 1)
    texts, labels = as_lists(sample)
    chosen = {}
    for variant, candidates in LR_CANDIDATES.items():
        best_lr = candidates[0]
        best_acc = -1
        for lr in candidates:
            _, _, history = train_model(
                texts,
                labels,
                val_texts,
                val_labels,
                variant=variant,
                epochs=EPOCHS[variant],
                learning_rate=lr,
                seed=1,
                output_dir="model_tmp",
            )
            acc = best_epoch(history)["val_accuracy"]
            print(f"lr check {variant} lr={lr}: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        chosen[variant] = best_lr
    return chosen


def run_baselines(sample, val_texts, val_labels):
    """Score both baselines trained on the sample."""
    texts, labels = as_lists(sample)
    common = most_common_label(labels)
    most_acc = accuracy_score(val_labels, [common] * len(val_texts))
    model = tfidf_model()
    model.fit(texts, labels)
    tfidf_acc = accuracy_score(val_labels, model.predict(val_texts))
    return most_acc, tfidf_acc


def main():
    os.makedirs("reports", exist_ok=True)
    train_df, val_df, _ = make_split()
    val_texts, val_labels = as_lists(val_df)

    learning_rates = choose_learning_rates(train_df, val_texts, val_labels)
    print("chosen learning rates:", learning_rates)

    rows = []
    history_rows = []
    for size in SIZES:
        for seed in SEEDS:
            sample = small_sample(train_df, size, seed)
            texts, labels = as_lists(sample)

            most_acc, tfidf_acc = run_baselines(sample, val_texts, val_labels)
            rows.append(
                {
                    "method": "most_common",
                    "size": size,
                    "seed": seed,
                    "val_accuracy": most_acc,
                }
            )
            rows.append(
                {
                    "method": "tfidf",
                    "size": size,
                    "seed": seed,
                    "val_accuracy": tfidf_acc,
                }
            )

            for variant in ["head", "full"]:
                _, _, history = train_model(
                    texts,
                    labels,
                    val_texts,
                    val_labels,
                    variant=variant,
                    epochs=EPOCHS[variant],
                    learning_rate=learning_rates[variant],
                    seed=seed,
                    output_dir="model_tmp",
                )
                best = best_epoch(history)
                rows.append(
                    {
                        "method": variant,
                        "size": size,
                        "seed": seed,
                        "val_accuracy": best["val_accuracy"],
                        "best_epoch": best["epoch"],
                    }
                )
                for entry in history:
                    history_rows.append(
                        {
                            "method": variant,
                            "size": size,
                            "seed": seed,
                            "epoch": entry["epoch"],
                            "train_loss": entry["train_loss"],
                            "val_accuracy": entry["val_accuracy"],
                        }
                    )
                print(f"{variant} size={size} seed={seed}: {best['val_accuracy']:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv("reports/results.csv", index=False)
    pd.DataFrame(history_rows).to_csv("reports/epoch_history.csv", index=False)
    print(results)

    summary = results.groupby(["method", "size"])["val_accuracy"].agg(
        ["mean", "min", "max"]
    )
    print(summary)

    plt.figure()
    for method in ["most_common", "tfidf", "head", "full"]:
        sub = results[results["method"] == method]
        grouped = sub.groupby("size")["val_accuracy"]
        means = grouped.mean()
        plt.plot(means.index, means.values, marker="o", label=method)
        plt.fill_between(
            means.index, grouped.min().values, grouped.max().values, alpha=0.2
        )
    plt.xlabel("Training rows")
    plt.ylabel("Validation accuracy")
    plt.title("Learning curve")
    plt.legend()
    plt.savefig("reports/learning_curve.png")
    print("wrote reports/results.csv and reports/learning_curve.png")


if __name__ == "__main__":
    main()
