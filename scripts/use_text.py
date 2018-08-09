def use_text_classification(file, model):    
    import pandas as pd
    test = pd.read_csv(file)
    test = test.dropna()
    test = pd.DataFrame(test.labeldata.str.split('\r\r\n').tolist(), index=test.labelname).stack()
    test = test.reset_index()[[0, 'labelname']] # var1 variable is currently labeled 0
    test.columns = ['labeldata', 'labelname'] # renaming var1
    
    import html
    test = html.unescape(test)
    # Shuffle data
    test = test.dropna()
    test = test[test['labeldata']!='']
    
    from empath import Empath
    lexicon = Empath()
    
    test_features = []
    for sentence in test['labeldata']:
        feature = lexicon.analyze(sentence, normalize=True)
        test_features.append(feature)
        
    test_features = pd.DataFrame(test_features)
    prediction = model.predict(test_features)   
    test['prediction'] = prediction
	
    return test