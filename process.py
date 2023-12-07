import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

def process_data(df):
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X[self.attribute_names].values
        
    num_attribs = df._get_numeric_data().columns
    cat_attribs = list(set(df.columns) - set(num_attribs))

    num_pipeline = Pipeline([
            ('select_df', DataFrameSelector(num_attribs)),
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])

    if cat_attribs != []:
        cat_pipeline = cat_pipeline = Pipeline([
                ('select_df', DataFrameSelector(cat_attribs)),
                ('categorical_enc', OneHotEncoder())
            ])
        full_pipeline = FeatureUnion(transformer_list=[
            ('num_pipe', num_pipeline),
            ('cat_pipe', cat_pipeline),
        ])
        transformed_data = full_pipeline.fit_transform(df)

        cat_encoder = cat_pipeline.named_steps["categorical_enc"]
        cat_one_hot_attribs = [cat for i in range(len(cat_attribs)) for cat in cat_encoder.categories_[i]]
        total_atribs = list(num_attribs) + cat_one_hot_attribs
        new_df = pd.DataFrame(transformed_data.toarray(), columns=total_atribs)

    else:
        transformed_data = num_pipeline.fit_transform(df)
        new_df = pd.DataFrame(transformed_data, columns=list(num_attribs))
    return new_df