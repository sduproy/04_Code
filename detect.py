#detect.py Features in, probability of fake out
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from config import CFG, set_seeds

FEATURE_PREFIXES = ("v_", "a_", "inh_") #inh = inherited features

def feature_cols(df):
    return [c for c in df.columns
            if c.startswith(FEATURE_PREFIXES)]

def train(df_train):
    set_seeds()
    cols = feature_cols(df_train)
    m = HistGradientBoostingClassifier(random_state=CFG.seed)
    m.fit(df_train[cols], (df_train["label"] == "fake").astype(int))
    return m, cols

def p_fake(model, cols, df) -> np.ndarray:
    return model.predict_proba(df[cols])[:, 1]