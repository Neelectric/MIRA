BRACKETS = [
    (0, 25000),
    (25000, 50000),
    (50000, 100000),
    (100000, None)
]
PERCENTAGES = [0, 5, 20, 30]

def tax_calculator(salary, brackets, percentages):
    total_tax = 0
    
    # write your code here
    for (start, end), percentage in zip(brackets, percentages):
        rate = percentage / 100
        if end is None:
            total_tax += rate * (salary - start)
            break
        difference = end - start
        if salary < start:
            break
        temp_salary = salary - start
        temp_salary = min(temp_salary, difference)
        total_tax += temp_salary * rate     
    return total_tax

some_salary = 80000

calculated = tax_calculator(salary=some_salary, 
               brackets=BRACKETS,
               percentages=PERCENTAGES)
print(calculated)