import pandas as pd
from difflib import SequenceMatcher

INPUT_FILE = 'PETs_Ukr.xlsx'
TRAIN_FILE = 'PETs_Ukr_Train.xlsx'
TEST_FILE  = 'PETs_Ukr_Test.xlsx'
SIMILARITY_THRESHOLD = 0.9 # flag pairs with similarity >= this value

# ── Split ────────────────────────────────────────────────────────────────────
# For each sheet:
#   1. Take N = min(count_label_0, count_label_1)
#   2. Randomly sample N rows from each label
#   3. Split each label's N rows exactly in half → N//2 train, N//2 test
# Result: every sheet has exactly the same number of pos and neg in both
# train and test, and train size == test size.

xls = pd.ExcelFile(INPUT_FILE)
train_sheets, test_sheets = {}, {}
total_dropped = 0

print('=== Splitting ===')
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)

    pos = df[df['label'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)
    neg = df[df['label'] == 0].sample(frac=1, random_state=42).reset_index(drop=True)

    N = min(len(pos), len(neg))
    half = N // 2  # each label contributes half to train, half to test
    dropped = len(df) - 4 * half

    train_pos = pos.iloc[:half]
    test_pos  = pos.iloc[half:2 * half]
    train_neg = neg.iloc[:half]
    test_neg  = neg.iloc[half:2 * half]

    train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    train_sheets[sheet] = train_df
    test_sheets[sheet]  = test_df
    total_dropped += dropped

    print(f'  {sheet:20s}  train={len(train_df):4d} (pos={half}, neg={half})'
          f'  test={len(test_df):4d} (pos={half}, neg={half})'
          f'  dropped={dropped}')

# Write output files
with pd.ExcelWriter(TRAIN_FILE, engine='openpyxl') as writer:
    for sheet, df in train_sheets.items():
        df.to_excel(writer, sheet_name=sheet, index=False)

with pd.ExcelWriter(TEST_FILE, engine='openpyxl') as writer:
    for sheet, df in test_sheets.items():
        df.to_excel(writer, sheet_name=sheet, index=False)

print(f'\nTotal rows dropped (odd extras): {total_dropped}')
print(f'Wrote {TRAIN_FILE} and {TEST_FILE}')

# ── Verification ─────────────────────────────────────────────────────────────

print('\n=== Verification ===')

total_exact   = 0
total_similar = 0

for sheet in xls.sheet_names:
    train_texts = train_sheets[sheet]['text'].dropna().astype(str).tolist()
    test_texts  = test_sheets[sheet]['text'].dropna().astype(str).tolist()

    train_set = set(train_texts)
    test_set  = set(test_texts)

    # Exact duplicates
    exact = train_set & test_set
    if exact:
        print(f'\n[{sheet}] {len(exact)} EXACT duplicate(s):')
        for t in list(exact)[:5]:
            print(f'  • {t[:120]}')
        if len(exact) > 5:
            print(f'  ... and {len(exact) - 5} more')

    # Near-duplicates via SequenceMatcher
    similar_pairs = []
    for tr_txt in train_texts:
        for te_txt in test_texts:
            if tr_txt == te_txt:
                continue  # already caught above
            ratio = SequenceMatcher(None, tr_txt, te_txt).ratio()
            if ratio >= SIMILARITY_THRESHOLD:
                similar_pairs.append((ratio, tr_txt, te_txt))

    similar_pairs.sort(reverse=True)

    if similar_pairs:
        print(f'\n[{sheet}] {len(similar_pairs)} near-duplicate pair(s) '
              f'(similarity >= {SIMILARITY_THRESHOLD}):')
        for ratio, tr_txt, te_txt in similar_pairs[:3]:
            print(f'  sim={ratio:.3f}')
            print(f'    TRAIN: {tr_txt[:100]}')
            print(f'    TEST:  {te_txt[:100]}')
        if len(similar_pairs) > 3:
            print(f'  ... and {len(similar_pairs) - 3} more pairs')

    total_exact   += len(exact)
    total_similar += len(similar_pairs)

print(f'\nTotal exact duplicates across all sheets : {total_exact}')
print(f'Total near-duplicates across all sheets  : {total_similar}')

if total_exact == 0 and total_similar == 0:
    print('\nPASS — no overlapping or highly similar sentences found.')
else:
    print('\nWARN — review flagged items above.')
