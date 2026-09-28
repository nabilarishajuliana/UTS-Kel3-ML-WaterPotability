import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def run_smote_pipeline(X_train, X_test, y_train, y_test):
    print("=" * 50)
    print("SMOTE + DECISION TREE")
    print("=" * 50)

    # -----------------------------------------------------------
    # 1. SMOTE PADA DATA TRAINING
    # -----------------------------------------------------------
    print("\nDistribusi sebelum SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nDistribusi setelah SMOTE:")
    print(y_train_smote.value_counts())

    # ------------------------------------------------------------------
    # 2. TRAIN DECISION TREE
    # ------------------------------------------------------------------
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_smote, y_train_smote)
    print("\nDecision Tree berhasil dilatih.")

    # -----------------------------------------------------------
    # 3. PREDIKSI DATA TESTING
    # -----------------------------------------------------------
    y_pred = model.predict(X_test)
    print(f"\nPrediksi selesai. Jumlah data prediksi: {len(y_pred)}")

    # -----------------------------------------------------------
    # 4.CONFUSION MATRIX
    # -----------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nHASIL EVALUASI")
    print("Confusion Matrix:\n", cm)
    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    return {
        "model": model, "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote, "confusion_matrix": cm,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1
    }

# ==========================================
# Visualisasi
# ==========================================
def visualize_smote_result(result, y_train, feature_names):
    folder = os.path.dirname(os.path.abspath(__file__))
    before = pd.Series(y_train).value_counts().sort_index()
    after = pd.Series(result["y_train_smote"]).value_counts().sort_index()

    # Distribusi
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(["Tidak Layak (0)", "Layak (1)"], before.values)
    ax[0].set_title("SEBELUM SMOTE (Imbalanced)")
    ax[1].bar(["Tidak Layak (0)", "Layak (1)"], after.values)
    ax[1].set_title("SESUDAH SMOTE (Balanced)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "viz_smote_1_distribusi.png"))
    plt.show()

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix — SMOTE + Decision Tree")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "viz_smote_2_confusion_matrix.png"))
    plt.show()

    # Metrik
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [result["accuracy"], result["precision"], result["recall"], result["f1_score"]]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(metrics, values)
    plt.ylim(0, 1.15)
    plt.title("Hasil Evaluasi SMOTE + Decision Tree")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.02,
                 f"{value:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "viz_smote_3_metrik.png"))
    plt.show()

    # Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(result["model"], feature_names=feature_names,
              class_names=["Tidak Layak", "Layak"],
              filled=True, rounded=True, max_depth=3)
    plt.title("Visualisasi Decision Tree (max_depth=3) — SMOTE")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "viz_smote_4_decision_tree.png"))
    plt.show()

# ==========================================
# TEST MANDIRI
# ==========================================
if __name__ == "__main__":
    from preprocessing import load_and_preprocess

    X_train, X_test, y_train, y_test = load_and_preprocess()
    result = run_smote_pipeline(X_train, X_test, y_train, y_test)
    visualize_smote_result(result, y_train, X_train.columns.tolist())