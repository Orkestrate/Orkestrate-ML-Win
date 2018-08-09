def train_text_classification(file):
    import pandas as pd
    train = pd.read_csv(file)
    train = train.dropna()
    train = pd.DataFrame(train.labeldata.str.split('\r\r\n').tolist(), index=train.labelname).stack()
    train = train.reset_index()[[0, 'labelname']] # var1 variable is currently labeled 0
    train.columns = ['labeldata', 'labelname'] # renaming var1
    import html
    train = html.unescape(train)
    # Shuffle data
    train = train.sample(frac=1, random_state=1).reset_index(drop=True)
    train = train.dropna()
    train = train[train['labeldata']!='']
    
    
	
    from empath import Empath
    lexicon = Empath()
    
    train_features = []
    for data in train['labeldata']:
        feature = lexicon.analyze(data, normalize=True)
        train_features.append(feature)
        
    train_features = pd.DataFrame(train_features)  
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='sag')
    
    model.fit(train_features,train['labelname'])

    return model