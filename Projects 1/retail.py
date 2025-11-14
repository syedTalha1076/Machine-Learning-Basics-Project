import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv('Retail_500.csv')

# Data cleaning
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

# User input
country = input("Enter country name (e.g., France): ")
min_support = float(input("Enter minimum support (e.g., 0.07): "))
min_lift = float(input("Enter minimum lift threshold (e.g., 1.0): "))

# Prepare basket
basket = (df[df['Country'] == country]
          .groupby(['InvoiceNo', 'Description'])['Quantity'].sum()
          .unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Encode quantities as 0/1
def encode_units(x):
    return 1 if x >= 1 else 0

basket_sets = basket.apply(lambda col: col.map(encode_units))  # <- fixed

# Apply Apriori
frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

if frequent_itemsets.empty:
    print("\nNo frequent itemsets found. Try lowering min_support or selecting another country.")
else:
    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

    if rules.empty:
        print("\nNo association rules found. Try lowering min_lift.")
    else:
        print("\nAssociation Rules:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        # User basket prediction
        user_items = input("\nEnter the items you already have (comma separated): ")
        user_items_set = set([item.strip() for item in user_items.split(',')])
        suggestions = rules[rules['antecedents'].apply(lambda x: user_items_set.issubset(x))]

        if not suggestions.empty:
            print("\nSuggested items based on your basket:")
            for idx, row in suggestions.iterrows():
                print(f"If you have {set(row['antecedents'])}, you might also want {set(row['consequents'])} (Confidence: {row['confidence']:.2f})")
        else:
            print("No suggestions found for the items you entered.")
