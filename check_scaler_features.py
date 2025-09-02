import pickle

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

if hasattr(scaler, 'feature_names_in_'):
    print('Scaler feature names:')
    print(list(scaler.feature_names_in_))
    print(f'Number of features: {len(scaler.feature_names_in_)}')
else:
    print('Scaler does not have feature_names_in_ attribute.')
    print('Scaler type:', type(scaler)) 