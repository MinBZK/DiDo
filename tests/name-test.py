radiation = 75.0
d = {'a': 1, 'b': 2}

def show_name(var):
    variable = [ i for i, j in locals().items() if j == var]
    print(variable)

show_name(d)