# ============================================
# MAIN - JANGAN DIEDIT
# ============================================
# File ini menjalankan seluruh pipeline:
# Preprocessing → SMOTE Pipeline → RUS Pipeline
# ============================================

from preprocessing import load_and_preprocess
from smote_pipeline import run_smote_pipeline
from rus_pipeline import run_rus_pipeline

if __name__ == "__main__":
    # Step 1: Preprocessing & Split Data (Adel)
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Step 2: SMOTE + Decision Tree (Juanne)
    run_smote_pipeline(X_train, X_test, y_train, y_test)

    # Step 3: RUS + Decision Tree (Risha)
    run_rus_pipeline(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 50)
    print("SELESAI - Bandingkan hasil SMOTE vs RUS di atas")
    print("=" * 50)