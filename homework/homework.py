import pandas as pd
import pickle
import gzip
import json
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

# --- Definición de Rutas ---
INPUT_DIR = "files/input"
MODELS_DIR = "files/models"
OUTPUT_DIR = "files/output"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Carga los datos intentando formatos zip o csv."""
    try:
        train = pd.read_csv(f"{INPUT_DIR}/train_data.csv.zip", compression="zip")
        test = pd.read_csv(f"{INPUT_DIR}/test_data.csv.zip", compression="zip")
    except FileNotFoundError:
        # Fallback por si están descomprimidos
        train = pd.read_csv(f"{INPUT_DIR}/train_data.csv")
        test = pd.read_csv(f"{INPUT_DIR}/test_data.csv")
    return train, test

def clean_data(df):
    """Realiza la limpieza especificada."""
    df = df.copy()
    # 1. Renombrar columna objetivo
    if 'default payment next month' in df.columns:
        df = df.rename(columns={'default payment next month': 'default'})
    
    # 2. Remover columna ID
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    # 3. Eliminar nulos
    df = df.dropna()
    
    # 4. Agrupar EDUCATION > 4 en la categoría 4 ('others')
    if 'EDUCATION' in df.columns:
        df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
        
    return df

# --- Flujo Principal de Ejecución ---

# 1. Cargar y Limpiar
train_df, test_df = load_data()
train_df = clean_data(train_df)
test_df = clean_data(test_df)

# 2. Separar X e y
x_train = train_df.drop(columns=['default'])
y_train = train_df['default']

x_test = test_df.drop(columns=['default'])
y_test = test_df['default']

# 3. Construir Pipeline
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 4. Optimización de Hiperparámetros (GridSearchCV)
param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [None],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1]
}

model = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring='balanced_accuracy',
    n_jobs=-1,
    refit=True
)

model.fit(x_train, y_train)

# 5. Guardar el Modelo Comprimido
with gzip.open(f"{MODELS_DIR}/model.pkl.gz", "wb") as f:
    pickle.dump(model, f)

# 6. Calcular Métricas
def get_metrics(dataset_name, y_true, y_pred):
    return {
        'type': 'metrics',  # Requerido por el test _test_metrics
        'dataset': dataset_name,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
    }

def get_confusion_matrix(dataset_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

metrics_data = [
    get_metrics('train', y_train, y_train_pred),
    get_metrics('test', y_test, y_test_pred),
    get_confusion_matrix('train', y_train, y_train_pred),
    get_confusion_matrix('test', y_test, y_test_pred)
]

# 7. Guardar Métricas en JSON
metrics_path = f"{OUTPUT_DIR}/metrics.json"
if os.path.exists(metrics_path):
    os.remove(metrics_path)

with open(metrics_path, "w") as f:
    for line in metrics_data:
        f.write(json.dumps(line) + "\n")