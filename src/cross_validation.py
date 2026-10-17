import pandas as pd
from sklearn.model_selection import cross_validate

def run_cv(models, X, y, cv, groups=None, scoring=("accuracy", "precision", "recall")):
    table = {}
    
    for name, estimator in models.items():
        s = cross_validate(estimator, X, y, cv=cv, groups=groups, scoring=list(scoring))

        table[name] = {m: f"{s[f'test_{m}'].mean():.3f} ± {s[f'test_{m}'].std():.3f}"
                       for m in scoring}
        
    return pd.DataFrame(table)