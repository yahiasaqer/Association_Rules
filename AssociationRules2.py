# get association rules by using FP-growth algorithm
import pyfpgrowth

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Milk', 'Unicorn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# use FP-growth to get patterns with minimum support = 3
patterns = pyfpgrowth.find_frequent_patterns(dataset, 3)
print('Patterns\n', patterns)

# use FP-growth to get association rules with minimum confidence = 0.7
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
print('Rules\n', rules)


