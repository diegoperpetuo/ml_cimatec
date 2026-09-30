import pandas as pd
import shap
from scipy import sparse

def prepare_data_for_shap(pipeline, X):
    """
    Prepara os dados de entrada para o SHAP.
    Recupera o pré-processador do pipeline e aplica a transformação no conjunto de dados.
    """

    preprocessor = pipeline.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(X)

    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    # Recupera os nomes das features
    feature_names = preprocessor.get_feature_names_out()

    # Converte para um DataFrame
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    return X_transformed


def create_explainer(pipeline):
    """
    Cria o objeto explainer do SHAP
    """

    # Recupera o modelo do pipeline
    model = pipeline.named_steps["model"]

    # Cria um explicador que entenda as previsões desse modelo
    explainer = shap.TreeExplainer(model)

    return explainer

def calculate_shap_values(explainer, X_transformed):
    """
    Calcula os valores SHAP para um conjunto de dados
    """

    # Calcula os valores SHAP para o conjunto de dados
    shap_values = explainer.shap_values(X_transformed)

    return shap_values