#detect.py Features in, probability of fake out
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from config import CFG, set_seeds
from sklearn.utils.class_weight import compute_sample_weight


FEATURE_PREFIXES = ("v_", "a_", "inh_") #inh = inherited features

def feature_cols(df):
    return [c for c in df.columns
            if c.startswith(FEATURE_PREFIXES)]

def train(df_train, balanced=False):
    set_seeds()
    cols = feature_cols(df_train)
    #This sw is a just for testing, pipeline is not changed as default is None anyways
    sw = (compute_sample_weight("balanced", (df_train["label"] == "fake").astype(int))
          if balanced else None)
    m = HistGradientBoostingClassifier(random_state=CFG.seed)
    m.fit(df_train[cols], (df_train["label"] == "fake").astype(int), sample_weight=sw)
    return m, cols

def p_fake(model, cols, df) -> np.ndarray:
    return model.predict_proba(df[cols])[:, 1]