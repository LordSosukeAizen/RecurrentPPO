def random_only_date(start, end):
    import pandas as pd
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return start + (end - start) * np.random.rand()