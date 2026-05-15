import pandas as pd


def examine_file(filepath):
    """Examine an Excel file and print statistics."""
    print(f'\n{"=" * 60}')
    print(f'Examining: {filepath}')
    print('=' * 60)
    xls = pd.ExcelFile(filepath)
    print('Sheet names:', xls.sheet_names)
    print('Total sheets:', len(xls.sheet_names))

    # Check first sheet structure
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    print(f'\nFirst sheet ({xls.sheet_names[0]}) structure:')
    print('Columns:', df.columns.tolist())
    print('Shape:', df.shape)
    print('\nFirst few rows:')
    print(df.head(3))

    if 'label' in df.columns:
        print('\nLabel distribution:')
        print(df['label'].value_counts())

    # Check statistics for all sheets
    print('\n=== Sheet Statistics ===')
    for sheet_name in xls.sheet_names:
        try:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
            positive = negative = 0
            if 'label' in df_sheet.columns:
                positive = (df_sheet['label'] == 1).sum()
                negative = (df_sheet['label'] == 0).sum()

            print(f'{sheet_name}: {df_sheet.shape[0]} rows, positive: {positive}, negative: {negative}')
        except Exception as e:
            print(f'{sheet_name}: Error - {e}')


examine_file('PETs_Ukr_Train.xlsx')
examine_file('PETs_Ukr_Test.xlsx')
