import pandas as pd
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split

INPUT_FILE = 'PETs_Ukr.xlsx'
TRAIN_FILE = 'PETs_Ukr_Train.xlsx'
TEST_FILE  = 'PETs_Ukr_Test.xlsx'
SPLIT_RATIO = 0.5          # 50% train, 50% test
SIMILARITY_THRESHOLD = 0.9 # flag pairs with similarity >= this value

# ── Split ────────────────────────────────────────────────────────────────────

xls = pd.ExcelFile(INPUT_FILE)
train_sheets, test_sheets = {}, {}

print('=== Splitting ===')
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)

    if 'label' in df.columns:
        # Stratified split to preserve label balance
        train_df, test_df = train_test_split(
            df, test_size=SPLIT_RATIO, stratify=df['label'], random_state=42
        )
    else:
        train_df, test_df = train_test_split(
            df, test_size=SPLIT_RATIO, random_state=42
        )

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    train_sheets[sheet] = train_df
    test_sheets[sheet]  = test_df

    pos_tr = (train_df['label'] == 1).sum() if 'label' in train_df.columns else '-'
    neg_tr = (train_df['label'] == 0).sum() if 'label' in train_df.columns else '-'
    pos_te = (test_df['label'] == 1).sum()  if 'label' in test_df.columns  else '-'
    neg_te = (test_df['label'] == 0).sum()  if 'label' in test_df.columns  else '-'
    print(f'  {sheet:20s}  train={len(train_df):4d} (pos={pos_tr}, neg={neg_tr})'
          f'  test={len(test_df):4d} (pos={pos_te}, neg={neg_te})')

# Write output files
with pd.ExcelWriter(TRAIN_FILE, engine='openpyxl') as writer:
    for sheet, df in train_sheets.items():
        df.to_excel(writer, sheet_name=sheet, index=False)

with pd.ExcelWriter(TEST_FILE, engine='openpyxl') as writer:
    for sheet, df in test_sheets.items():
        df.to_excel(writer, sheet_name=sheet, index=False)

print(f'\nWrote {TRAIN_FILE} and {TEST_FILE}')

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
