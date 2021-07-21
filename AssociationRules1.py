import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Milk', 'Unicorn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# TransactionEncoder is a model to deal with transactions
te = TransactionEncoder()

# Transform the transaction dataset to binary 2D array
te_ary = te.fit(dataset).transform(dataset)
print(te_ary)

# convert the array of transaction data array into pandas DataFrame
df = pd.DataFrame(te_ary, columns=te.columns_)

# get the frequent itemsets by using apriori algorithm
frequentItemsets = apriori(df, min_support=0.6, use_colnames=True)
print('Itemsets\n', frequentItemsets)

#get the association rules
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequentItemsets, min_threshold=0.7) #min_threshold is the minimum confidence
print('Rules\n', rules)
