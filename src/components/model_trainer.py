import os,sys 
import mlflow
import mlflow.sklearn

from src.exception.exception import NetWorkSecurityException
from src.logging.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig

from src.utils.main_utils.utils import save_object, load_object
from src.utils.main_utils.utils import load_numpy_array_data
from src.utils.ml_utils.model.estimator import NetworkModel
from src.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise NetWorkSecurityException(e, sys)
    def track_mlflow(self, best_model, classificationmetric):
            with mlflow.start_run():
                f1 = classificationmetric.f1_score
                precision = classificationmetric.precision_score
                recall = classificationmetric.recall_score
                
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall_score", recall)
                mlflow.sklearn.log_model(best_model, "model")

        
    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}
            best_model = None
            best_score = -1
            
            for name, model in models.items():
                gs = GridSearchCV(model, params[name], cv=3, verbose=2)
                gs.fit(X_train, y_train)

                model = gs.best_estimator_
                y_pred = model.predict(X_test)

                score = f1_score(y_test, y_pred)

                report[name] = score

                if score > best_score:
                    best_score = score
                    best_model = model

            return report, best_model
        
        except Exception as e:
            raise NetWorkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
        }
        
        params = {
            "Decision Tree": {
                "criterion": ['gini', 'entropy', 'log_loss'],
                # "splitter": ['best', 'random'],
                # 'max_features': ['sqrt', 'log2'],
            },
            "Random Forest": {
                # 'criterion': ['gini', 'entropy', 'log_loss'],
                # 'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                # 'loss': ['log_loss', 'experimental'],
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                # 'criterion': ['squared_error', 'friedman_mse'],
                # 'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {
                "C": [0.1, 1, 10]},
            "AdaBoost": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256],
            }
        }
        model_report, best_model = self.evaluate_models(
                                 X_train, y_train, X_test, y_test, models, params
                            )

        # to get model score from dict
        best_model_score = max(model_report.values())
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = best_model
        y_train_pred = best_model.predict(X_train)
        classification_train_metric=get_classification_score(y_train, y_train_pred)

        # track mlflow
        self.track_mlflow(best_model, classification_train_metric)
        
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_test, y_test_pred)
        
        self.track_mlflow(best_model, classification_test_metric)
        
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)
        
        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
        save_object("final_model/model.pkl",best_model)

        
        # Model trainer artifact
        model_trainer_artifact = ModelTrainerArtifact(self.model_trainer_config.trained_model_file_path, classification_train_metric, classification_test_metric)
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
            
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact
            
            
        except Exception as e:
            raise NetWorkSecurityException(e, sys)    