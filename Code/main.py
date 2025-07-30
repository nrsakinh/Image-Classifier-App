import sys
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score
import cv2
import pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                             QComboBox, QTabWidget, QTextEdit, QProgressBar, 
                             QMessageBox, QSplitter, QFrame, QGridLayout, 
                             QGroupBox, QStackedWidget, QDialogButtonBox,
                             QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize


class DataPreparationWorker(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    completed = pyqtSignal(object, object, object, object, object)
    
    def __init__(self, dataset_type, dataset_path):
        super().__init__()
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        
    def run(self):
        try:
            if self.dataset_type == "Iris":
                self.prepare_iris_dataset()
            else:
                self.prepare_rose_dataset()
        except Exception as e:
            self.status_update.emit(f"Error in data preparation: {str(e)}")
    
    def prepare_iris_dataset(self):
        self.status_update.emit("Loading Iris dataset...")
        self.progress_update.emit(10)
    
        df = pd.read_csv(self.dataset_path)
        self.progress_update.emit(30)
        self.status_update.emit("Dataset loaded successfully")

        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
        elif df.columns[0] == 'Unnamed: 0' or df.columns[0].lower() == 'id':
            df = df.drop(df.columns[0], axis=1)
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        if isinstance(y[0], str) and y[0].startswith('Iris-'):
            y = np.array([species.replace('Iris-', '') for species in y])
        
        unique_labels = np.unique(y)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y_numeric = np.array([label_mapping[label] for label in y])
        
        self.progress_update.emit(50)
        self.status_update.emit("Preparing train-test split...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.progress_update.emit(80)
        self.status_update.emit("Data preparation completed")
        
        feature_names = list(df.columns)[:-1]  
        
        self.progress_update.emit(100)
        self.status_update.emit("Data preparation completed successfully!")
        
        metadata = {
            'feature_names': feature_names,
            'label_names': unique_labels,
            'dataset_type': 'Iris',
            'scaler': scaler,
            'label_mapping': label_mapping  
        }
        
        self.completed.emit(X_train, y_train, X_test, y_test, metadata)
        
    def augment_image(img):
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        # Random vertical flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
        # Random rotation
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
        # Random crop and resize
        crop_scale = random.uniform(0.8, 1.0)
        h, w = img.shape[:2]
        ch, cw = int(h * crop_scale), int(w * crop_scale)
        y = random.randint(0, h - ch)
        x = random.randint(0, w - cw)
        img = img[y:y+ch, x:x+cw]
        img = cv2.resize(img, (w, h))
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        # Random contrast
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            mean = np.mean(img, axis=(0,1), keepdims=True)
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
        # Random noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 10, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def extract_color_features(img):
        """Extract enhanced color histogram features from an image"""
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        h_hist = cv2.calcHist([hsv_img], [0], None, [32], [0, 180])  
        s_hist = cv2.calcHist([hsv_img], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv_img], [2], None, [16], [0, 256])
        
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        pixels = hsv_img.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        k = 5  
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        _, counts = np.unique(labels, return_counts=True)
        percentages = counts / sum(counts)
        
        color_features = []
        for i in range(k):
            color_features.append(percentages[i])
            color_features.extend(centers[i])
        
        combined_features = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten(), color_features])
        return combined_features

    def prepare_rose_dataset(self):
        self.status_update.emit("Loading Rose dataset...")
        self.progress_update.emit(10)

        image_size = (64, 64)
        X_images = []
        X_color_features = []
        y = []
        labels = []
        pca_used = False
        pca_object = None

        subdirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]

        if subdirs:
            self.status_update.emit("Detected folder structure for classes")
            for i, subdir in enumerate(subdirs):
                class_path = os.path.join(self.dataset_path, subdir)
                self.status_update.emit(f"Processing class: {subdir}")
                labels.append(subdir)

                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, filename)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img_resized = cv2.resize(img, image_size)

                                color_features = DataPreparationWorker.extract_color_features(img_resized)

                                X_images.append(img_resized.flatten())
                                X_color_features.append(color_features)
                                y.append(i)
                        except Exception as e:
                            self.status_update.emit(f"Error processing image {img_path}: {str(e)}")

                progress = int(30 + (i / len(subdirs)) * 40)
                self.progress_update.emit(progress)
        else:
            pass

        X_images = np.array(X_images)
        X_color_features = np.array(X_color_features)
        y = np.array(y)

        self.status_update.emit(f"Dataset loaded: {len(X_images)} images with {len(np.unique(y))} classes")
        self.progress_update.emit(70)

        # --- AUGMENTATION TO 250 PER CLASS ---
        target_samples_per_class = 250
        aug_X_images = []
        aug_X_color_features = []
        aug_y = []

        unique_classes = np.unique(y)
        for cls in unique_classes:
            indices = np.where(y == cls)[0]
            n_current = len(indices)
            imgs_cls = [X_images[i].reshape(image_size[0], image_size[1], 3) for i in indices]
            feats_cls = [X_color_features[i] for i in indices]
            # Add originals
            for i in indices:
                aug_X_images.append(X_images[i])
                aug_X_color_features.append(X_color_features[i])
                aug_y.append(cls)
            # Augment until 700 per class
            for _ in range(target_samples_per_class - n_current):
                idx = np.random.choice(range(n_current))
                img = imgs_cls[idx]
                aug_img = DataPreparationWorker.augment_image(img)
                aug_feat = DataPreparationWorker.extract_color_features(aug_img)
                aug_X_images.append(aug_img.flatten())
                aug_X_color_features.append(aug_feat)
                aug_y.append(cls)

        X_images = np.array(aug_X_images)
        X_color_features = np.array(aug_X_color_features)
        y = np.array(aug_y)

        # --- END AUGMENTATION ---

        # Combine features (with or without PCA)
        if X_images.shape[1] > 1000:
            self.status_update.emit("Applying PCA for dimensionality reduction...")
            from sklearn.decomposition import PCA

            n_samples = X_images.shape[0]
            n_components = min(100, n_samples-1, int(0.75*X_images.shape[1]))
            self.status_update.emit(f"Using {n_components} components for PCA based on dataset size")

            pca = PCA(n_components=n_components)
            X_images_pca = pca.fit_transform(X_images)
            self.status_update.emit(f"Reduced features from {image_size[0]*image_size[1]*3} to {X_images_pca.shape[1]}")
            self.progress_update.emit(80)
            pca_used = True
            pca_object = pca

            color_weight = 2.0
            weighted_color_features = X_color_features * color_weight

            X = np.hstack((X_images_pca, weighted_color_features))
            self.status_update.emit(f"Applied color weighting (x{color_weight}) to enhance color-based classification")
        else:
            color_weight = 2.0
            weighted_color_features = X_color_features * color_weight
            X = np.hstack((X_images, weighted_color_features))
            self.status_update.emit(f"Applied color weighting (x{color_weight}) to enhance color-based classification")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.progress_update.emit(90)

        metadata = {
            'feature_names': None,
            'label_names': labels,
            'dataset_type': 'Rose',
            'scaler': scaler,
            'image_size': image_size,
            'n_color_features': X_color_features.shape[1],
            'color_features': True,
            'color_weight': color_weight,
        }

        if pca_used and pca_object is not None:
            metadata['pca'] = pca_object
            metadata['n_pca_components'] = pca_object.n_components_
            metadata['original_image_dim'] = X_images.shape[1]

        self.progress_update.emit(100)
        self.status_update.emit("Data preparation completed successfully!")

        self.completed.emit(X_train, y_train, X_test,y_test, metadata)

class ModelTrainingWorker(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    completed = pyqtSignal(object, object)
    
    def __init__(self, X_train, y_train, optimization=False):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.optimization = optimization
        
    def run(self):
        try:
            if self.optimization:
                self.train_optimized_model()
            else:
                self.train_basic_model()
        except Exception as e:
            self.status_update.emit(f"Error in model training: {str(e)}")
    
    def train_basic_model(self):
        self.status_update.emit("Training SVM model...")
        self.progress_update.emit(10)
        
        model = svm.SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42  
        )
        
        self.progress_update.emit(30)
        self.status_update.emit("Fitting the model...")
        
        model.fit(self.X_train, self.y_train)
        
        self.progress_update.emit(90)
        self.status_update.emit("Model training completed")
        
        model_info = {
            'name': 'SVM (RBF kernel)',
            'params': model.get_params(),
            'optimized': False
        }
        
        self.progress_update.emit(100)
        self.status_update.emit("Model successfully trained!")
        
        self.completed.emit(model, model_info)
        
    def train_optimized_model(self):
        self.status_update.emit("Starting hyperparameter optimization for SVM...")
        self.progress_update.emit(10)
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly'],
            'class_weight': ['balanced'] 
        }
        
        from sklearn.model_selection import StratifiedKFold
        n_splits = min(3, len(np.unique(self.y_train)))  
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        self.status_update.emit("Performing grid search cross-validation...")
        self.progress_update.emit(20)
        
        from sklearn.metrics import make_scorer, f1_score
        scorer = make_scorer(f1_score, average='weighted')
        
        grid_search = GridSearchCV(
            estimator=svm.SVC(probability=True),
            param_grid=param_grid,
            cv=cv,
            scoring=scorer,
            verbose=1,
            n_jobs=-1
        )
        
        self.status_update.emit("Training models with different hyperparameters...")
        grid_search.fit(self.X_train, self.y_train)
                
        self.progress_update.emit(80)
        self.status_update.emit(f"Hyperparameter optimization completed. Best score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
        
        model_info = {
            'name': 'Optimized SVM',
            'params': model.get_params(),
            'optimized': True,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'params': [str(p) for p in grid_search.cv_results_['params']]
            }
        }
        
        optimization_details = "Top 5 parameter combinations:\n\n"
        mean_scores = grid_search.cv_results_['mean_test_score']
        params = grid_search.cv_results_['params']
        top_indices = np.argsort(mean_scores)[-5:][::-1]  
        
        for i, idx in enumerate(top_indices):
            optimization_details += f"{i+1}. Score: {mean_scores[idx]:.4f}\n"
            optimization_details += f"   Parameters: C={params[idx]['C']}, "
            optimization_details += f"gamma={params[idx]['gamma']}, "
            optimization_details += f"kernel={params[idx]['kernel']}\n\n"
        
        self.status_update.emit(f"Best parameters: {grid_search.best_params_}")
        self.progress_update.emit(90)
        
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            test_score = model.score(self.X_test, self.y_test)
            self.status_update.emit(f"Test accuracy with optimized model: {test_score:.4f}")
        
        self.progress_update.emit(100)
        self.status_update.emit(f"Optimized model successfully trained! Best parameters: {grid_search.best_params_}")
        
        self.completed.emit(model, model_info)
        
class CustomFigureCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setMinimumSize(250, 200)

class ImageClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classification System")
        self.setGeometry(100, 100, 1200, 800)
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.metadata = None
        self.model = None
        self.model_info = None
        self.evaluation_results = None
        self.current_image = None
        self.current_image_path = None
        self.setup_ui()
        
    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(300)
        
        workflow_group = QGroupBox("Workflow Steps")
        workflow_layout = QVBoxLayout()
        
        data_prep_group = QGroupBox("1. Data Preparation && Processing")
        data_prep_layout = QVBoxLayout()
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Iris Dataset", "Rose Dataset"])
        
        self.dataset_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #ccc;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #ccc;
            }
            QComboBox QAbstractItemView::item {
                min-height: 25px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #2185d0;
                color: #000000 !important;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #2185d0;
                color: #000000;
            }
        """)
        
        self.dataset_path_label = QLabel("Dataset Path: Not selected")
        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_dataset_btn.clicked.connect(self.load_dataset)
        self.load_dataset_btn.setStyleSheet("background-color: #2185d0; color: white;")
        
        data_prep_layout.addWidget(self.dataset_combo)
        data_prep_layout.addWidget(self.dataset_path_label)
        data_prep_layout.addWidget(self.load_dataset_btn)
        data_prep_group.setLayout(data_prep_layout)
        
        model_group = QGroupBox("2. Training && Optimization")
        model_layout = QVBoxLayout()
        
        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.train_model)
        self.train_model_btn.setEnabled(False)
        self.train_model_btn.setStyleSheet("background-color: #21ba45; color: white;")
        
        self.optimize_model_btn = QPushButton("Optimize Model")
        self.optimize_model_btn.clicked.connect(self.optimize_model)
        self.optimize_model_btn.setEnabled(False)
        self.optimize_model_btn.setStyleSheet("background-color: #21ba45; color: white;")
        
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        
        model_layout.addWidget(self.train_model_btn)
        model_layout.addWidget(self.optimize_model_btn)
        model_layout.addWidget(QLabel(""))
        model_layout.addWidget(self.save_model_btn)
        model_layout.addWidget(self.load_model_btn)
        model_group.setLayout(model_layout)
        
        testing_group = QGroupBox("3. Testing")
        testing_layout = QVBoxLayout()
        
        self.evaluate_model_btn = QPushButton("Evaluate Model")
        self.evaluate_model_btn.clicked.connect(self.evaluate_model)
        self.evaluate_model_btn.setEnabled(False)
        self.evaluate_model_btn.setStyleSheet("background-color: #f2711c; color: white;")
        
        testing_layout.addWidget(self.evaluate_model_btn)
        testing_group.setLayout(testing_layout)
        
        prediction_group = QGroupBox("4. Display Output")
        prediction_layout = QVBoxLayout()
        
        self.load_image_btn = QPushButton("Load New Image / Values")
        self.load_image_btn.clicked.connect(self.load_new_image)
        self.load_image_btn.setEnabled(False)
        self.load_image_btn.setStyleSheet("background-color: #6435c9; color: white;")
        
        self.verify_image_btn = QPushButton("Verify Image / Values")
        self.verify_image_btn.clicked.connect(self.classify_image)
        self.verify_image_btn.setEnabled(False)
        self.verify_image_btn.setStyleSheet("background-color: #6435c9; color: white;")
        
        self.gradcam_btn = QPushButton("Apply Grad-CAM Visualization")
        self.gradcam_btn.clicked.connect(self.apply_gradcam)
        self.gradcam_btn.setEnabled(False)
        self.gradcam_btn.setStyleSheet("background-color: #6435c9; color: white;")
        
        prediction_layout.addWidget(self.load_image_btn)
        prediction_layout.addWidget(self.verify_image_btn)
        prediction_layout.addWidget(self.gradcam_btn)
        prediction_group.setLayout(prediction_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-style: italic;")
        
        self.help_btn = QPushButton("Info / Help")
        self.help_btn.clicked.connect(self.show_help)
        self.help_btn.setStyleSheet("background-color: #767676; color: white;")
        
        workflow_layout.addWidget(data_prep_group)
        workflow_layout.addWidget(model_group)
        workflow_layout.addWidget(testing_group)
        workflow_layout.addWidget(prediction_group)
        workflow_group.setLayout(workflow_layout)
        
        left_layout.addWidget(workflow_group)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.help_btn)
        
        self.tab_widget = QTabWidget()
        
        self.data_tab = QWidget()
        data_tab_layout = QVBoxLayout()
        
        self.data_canvas_container = QVBoxLayout()
        self.data_canvas = CustomFigureCanvas()
        self.data_canvas_container.addWidget(self.data_canvas)
        
        data_info_layout = QHBoxLayout()
        self.data_info_text = QTextEdit()
        self.data_info_text.setReadOnly(True)
        data_info_layout.addWidget(self.data_info_text)
        
        data_tab_layout.addLayout(self.data_canvas_container)
        data_tab_layout.addLayout(data_info_layout)
        self.data_tab.setLayout(data_tab_layout)
        
        self.eval_tab = QWidget()
        eval_tab_layout = QVBoxLayout()
        
        eval_splitter = QSplitter(Qt.Vertical)
        
        eval_top_widget = QWidget()
        eval_top_layout = QHBoxLayout()
        
        self.conf_matrix_canvas = CustomFigureCanvas()
        
        eval_metrics_widget = QWidget()
        eval_metrics_layout = QVBoxLayout()
        self.eval_metrics_text = QTextEdit()
        self.eval_metrics_text.setReadOnly(True)
        eval_metrics_layout.addWidget(QLabel("Classification Metrics"))
        eval_metrics_layout.addWidget(self.eval_metrics_text)
        eval_metrics_widget.setLayout(eval_metrics_layout)
        
        eval_top_layout.addWidget(self.conf_matrix_canvas, 60)
        eval_top_layout.addWidget(eval_metrics_widget, 40)
        eval_top_widget.setLayout(eval_top_layout)
        
        eval_bottom_widget = QWidget()
        eval_bottom_layout = QGridLayout()
        
        self.precision_recall_canvas = CustomFigureCanvas()
        self.accuracy_canvas = CustomFigureCanvas()
        
        eval_bottom_layout.addWidget(QLabel("Precision-Recall"), 0, 0)
        eval_bottom_layout.addWidget(self.precision_recall_canvas, 1, 0)
        eval_bottom_layout.addWidget(QLabel("Accuracy"), 0, 1)
        eval_bottom_layout.addWidget(self.accuracy_canvas, 1, 1)
        
        eval_bottom_widget.setLayout(eval_bottom_layout)
        
        eval_splitter.addWidget(eval_top_widget)
        eval_splitter.addWidget(eval_bottom_widget)
        
        eval_tab_layout.addWidget(eval_splitter)
        self.eval_tab.setLayout(eval_tab_layout)
        
        self.pred_tab = QWidget()
        pred_tab_layout = QVBoxLayout()
        
        pred_display_layout = QHBoxLayout()
        
        image_group = QGroupBox("Input Image")
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")
        image_layout.addWidget(self.image_label)
        image_group.setLayout(image_layout)
        
        results_group = QGroupBox("Classification Results")
        results_layout = QVBoxLayout()
        self.pred_result_text = QTextEdit()
        self.pred_result_text.setReadOnly(True)
        self.pred_result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pred_result_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  
        self.pred_result_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  
        self.pred_result_text.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.pred_result_text)
        results_group.setLayout(results_layout)
        
        pred_display_layout.addWidget(image_group)
        pred_display_layout.addWidget(results_group)
        
        pred_tab_layout.addLayout(pred_display_layout)
        self.pred_tab.setLayout(pred_tab_layout)
        
        self.tab_widget.addTab(self.data_tab, "Data Visualization")
        self.tab_widget.addTab(self.eval_tab, "Model Evaluation")
        self.tab_widget.addTab(self.pred_tab, "Testing")
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.tab_widget, 1)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def apply_styling(self):
        QApplication.setStyle("Fusion")
        
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f8f8f8;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px 10px;
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #aaa;
            }
            QComboBox {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff; /* Background color for dropdown */
                color: #000000; /* Default text color */
            }

            QComboBox QAbstractItemView::item:hover {
                background-color: #b3d1ff; /* Lighter blue background */
                color: #000000; /* Black text */
            }
            QLabel {
                color: #333;
            }
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #2185d0;
                width: 1px;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #ccc;
                border-bottom-color: #ddd;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #f8f8f8;
                border-bottom-color: #f8f8f8;
            }
        """)
    
    def load_dataset(self):
        dataset_type = self.dataset_combo.currentText().split()[0]  
        
        if dataset_type == "Iris":
            filepath, _ = QFileDialog.getOpenFileName(self, "Select Iris Dataset", "", "CSV Files (*.csv)")
        else:  # Rose
            filepath = QFileDialog.getExistingDirectory(self, "Select Rose Dataset Directory")
        
        if filepath:
            self.dataset_path_label.setText(f"Dataset Path: {os.path.basename(filepath)}")
            
            self.set_status(f"Preparing {dataset_type} dataset...")
            self.data_preparation_worker = DataPreparationWorker(dataset_type, filepath)
            self.data_preparation_worker.progress_update.connect(self.update_progress)
            self.data_preparation_worker.status_update.connect(self.set_status)
            self.data_preparation_worker.completed.connect(self.on_data_preparation_complete)
            self.data_preparation_worker.start()
            
            self.load_dataset_btn.setEnabled(False)
            self.dataset_combo.setEnabled(False)
    
    def on_data_preparation_complete(self, X_train, y_train, X_test, y_test, metadata):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metadata = metadata
        
        self.set_status("Dataset preparation complete. Ready for training.")
        self.train_model_btn.setEnabled(True)
        self.optimize_model_btn.setEnabled(True)
        self.load_dataset_btn.setEnabled(True)
        self.dataset_combo.setEnabled(True)
        self.update_progress(100)
        self.visualize_data()
        self.display_dataset_info()
        self.tab_widget.setCurrentIndex(0)
    
    def visualize_data(self):
        self.data_canvas.fig.clear()
        
        if self.metadata['dataset_type'] == 'Iris':
            self.visualize_iris_data()
        else:
            self.visualize_rose_data()
        
        self.data_canvas.draw()
    
    def visualize_iris_data(self):
        indices = np.random.choice(len(self.X_train), min(100, len(self.X_train)), replace=False)
        X_sample = self.X_train[indices]
        y_sample = self.y_train[indices]

        feature_names = self.metadata['feature_names'] if self.metadata['feature_names'] else [f'Feature {i}' for i in range(X_sample.shape[1])]

        ax = self.data_canvas.fig.add_subplot(111)
        
        if X_sample.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_sample)
            
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='viridis', alpha=0.8)
            ax.set_title('PCA visualization of Iris Data')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            
            if self.metadata['label_names'] is not None:
                legend_labels = self.metadata['label_names']
                handles, _ = scatter.legend_elements()
                ax.legend(handles, legend_labels, title="Classes")
        else:
            ax.scatter(X_sample[:, 0], X_sample[:, 1], c=y_sample, cmap='viridis', alpha=0.8)
            ax.set_title('Visualization of Iris Data')
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            
    def extract_dominant_colors(self, img, k=3):
        """Extract dominant colors from an image"""
        pixels = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Apply KMeans
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count labels to find most dominant colors
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort by counts
        indices = np.argsort(counts)[::-1]
        centers = centers[indices]
        counts = counts[indices]
        
        # Normalize counts to get percentages
        percentages = counts / sum(counts)
        
        # Convert centers to uint8 for color display
        centers_uint8 = centers.astype(np.uint8)
        
        # Return centers and their percentages
        return centers_uint8, percentages
    
    def apply_gradcam(self):
        if self.model is None or self.current_image is None or self.metadata['dataset_type'] != 'Rose':
            QMessageBox.warning(self, "Error", "Grad-CAM requires a trained model and a loaded image.\nOnly works with Rose dataset.")
            return
        
        try:
            import io
            import base64
            from scipy.ndimage import gaussian_filter
            
            self.set_status("Applying Grad-CAM visualization...")
            
            # Read the original image
            img = cv2.imread(self.current_image_path)
            if img is None:
                raise ValueError("Failed to load image for Grad-CAM")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_size = self.metadata.get('image_size', (64, 64))
            img_resized = cv2.resize(img, image_size)
            img_original = cv2.resize(img, (256, 256))
            
            # Create feature contribution map for SVM
            # Since SVM doesn't natively support Grad-CAM, we'll implement a gradient-free approach
            # based on feature sensitivity
            
            # Extract features
            color_features = DataPreparationWorker.extract_color_features(img_resized)
            img_features = img_resized.flatten()
            
            # Apply PCA if needed
            if 'pca' in self.metadata and self.metadata['pca'] is not None:
                pca = self.metadata['pca']
                img_features = pca.transform([img_features])[0]
            
            # Combine features
            color_weight = self.metadata.get('color_weight', 2.0)
            weighted_color_features = color_features * color_weight
            X = np.hstack(([img_features], [weighted_color_features]))
            
            # Scale features
            if 'scaler' in self.metadata and self.metadata['scaler'] is not None:
                X = self.metadata['scaler'].transform(X)
            
            # Get model prediction
            y_pred = self.model.predict(X)[0]
            
            # Get class name
            if 'label_names' in self.metadata and len(self.metadata['label_names']) > 0:
                pred_class_name = self.metadata['label_names'][y_pred]
            else:
                pred_class_name = f"Class {y_pred}"
            
            # For SVM, compute feature importance through permutation
            importance_map = np.zeros((image_size[0], image_size[1]))
            
            # Get the decision function (distance to hyperplane)
            base_score = self.model.decision_function(X)[0]
            if hasattr(self.model, 'classes_') and len(self.model.classes_) > 2:
                # For multi-class, get the score for the predicted class
                base_score = base_score[y_pred]
            
            # Compute pixel-wise importance by region perturbation
            # Use smaller blocks for finer detail
            block_size = 4  # Reduced from 8 to 4 for finer detail
            for i in range(0, image_size[0], block_size):
                for j in range(0, image_size[1], block_size):
                    # Create a perturbed version of the image
                    img_perturbed = img_resized.copy()
                    
                    # Apply a significant blur to this region
                    i_end = min(i + block_size, image_size[0])
                    j_end = min(j + block_size, image_size[1])
                    img_perturbed[i:i_end, j:j_end] = np.mean(img_perturbed[i:i_end, j:j_end], axis=(0, 1))
                    
                    # Extract features for the perturbed image
                    color_features_perturbed = DataPreparationWorker.extract_color_features(img_perturbed)
                    img_features_perturbed = img_perturbed.flatten()
                    
                    # Apply PCA if needed
                    if 'pca' in self.metadata and self.metadata['pca'] is not None:
                        img_features_perturbed = pca.transform([img_features_perturbed])[0]
                    
                    # Combine features
                    weighted_color_features_perturbed = color_features_perturbed * color_weight
                    X_perturbed = np.hstack(([img_features_perturbed], [weighted_color_features_perturbed]))
                    
                    # Scale features
                    if 'scaler' in self.metadata and self.metadata['scaler'] is not None:
                        X_perturbed = self.metadata['scaler'].transform(X_perturbed)
                    
                    # Get perturbed score
                    perturbed_score = self.model.decision_function(X_perturbed)[0]
                    if hasattr(self.model, 'classes_') and len(self.model.classes_) > 2:
                        perturbed_score = perturbed_score[y_pred]
                    
                    # Compute importance: how much the score changes when this region is perturbed
                    importance = abs(base_score - perturbed_score)
                    importance_map[i:i_end, j:j_end] = importance
            
            # Normalize importance map
            if np.max(importance_map) > 0:
                importance_map = (importance_map - np.min(importance_map)) / (np.max(importance_map) - np.min(importance_map))
            
            # Apply stronger gaussian smoothing for a smoother heatmap
            importance_map = gaussian_filter(importance_map, sigma=1.5)
            
            # Resize importance map to the size of the original image using BICUBIC interpolation
            importance_map_resized = cv2.resize(importance_map, (256, 256), 
                                                interpolation=cv2.INTER_CUBIC)
            
            # Further refine with another gaussian blur after resizing
            importance_map_resized = gaussian_filter(importance_map_resized, sigma=0.7)
            
            # Create heatmap with Jet colormap
            heatmap = np.uint8(255 * importance_map_resized)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Create different alpha blending levels for improved visualization
            alpha_strong = 0.7  # More transparency to see the original image better
            alpha_medium = 0.5  # Balanced blend
            alpha_light = 0.3   # Light overlay to see more details
            
            # Create 3 versions of the blended image with different transparency levels
            superimposed_strong = heatmap * alpha_strong + img_original * (1 - alpha_strong)
            superimposed_medium = heatmap * alpha_medium + img_original * (1 - alpha_medium)
            superimposed_light = heatmap * alpha_light + img_original * (1 - alpha_light)
            
            # Convert to uint8
            superimposed_strong = np.uint8(superimposed_strong)
            superimposed_medium = np.uint8(superimposed_medium)
            superimposed_light = np.uint8(superimposed_light)
            
            # Display the Grad-CAM visualization - using the medium blend as the main version
            self.visualize_gradcam(img_original, heatmap, superimposed_medium, pred_class_name, 
                                  superimposed_light, superimposed_strong)
            
            self.set_status(f"Grad-CAM visualization applied for class: {pred_class_name}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.set_status(f"Error during Grad-CAM visualization: {str(e)}")
    
    def visualize_gradcam(self, original_img, heatmap, superimposed_img, class_name, 
                     superimposed_light=None, superimposed_strong=None):
        """Display the Grad-CAM visualization results in the prediction tab"""
        import io
        import base64
        
        # Reduce figure size and adjust dpi for better fit in the UI container
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=100)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot original image
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image", fontsize=10)
        axes[0].axis('off')
        
        # Plot heatmap
        axes[1].imshow(heatmap)
        axes[1].set_title("Activation Heatmap", fontsize=10)
        axes[1].axis('off')
        
        # Plot standard superimposed image (medium blend)
        axes[2].imshow(superimposed_img)
        axes[2].set_title(f"Grad-CAM (50% blend)", fontsize=10)
        axes[2].axis('off')
        
        # If we have the light version, plot it, otherwise plot the strong version
        if superimposed_light is not None:
            axes[3].imshow(superimposed_light)
            axes[3].set_title(f"Grad-CAM (30% blend)", fontsize=10)
        else:
            axes[3].imshow(superimposed_strong)
            axes[3].set_title(f"Grad-CAM (70% blend)", fontsize=10)
        axes[3].axis('off')
        
        # Add a main title with smaller font size
        plt.suptitle(f"Grad-CAM Visualization for Class: {class_name}", fontsize=12)
        
        # Adjust spacing between subplots to be more compact
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Convert figure to image data with optimized quality
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Create HTML to display the results with responsive sizing
        result_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; color: #222; width: 100%;">
            <div style="margin-bottom: 8px;">
                <span style="font-size: 14px; font-weight: bold; color: #2185d0;">Grad-CAM Visualization</span>
            </div>
            <div style="margin-bottom: 8px;">
                <p>Visualizing model decision for class: <b>{class_name}</b></p>
                <p>The heatmap highlights image regions that most influenced the model's prediction.</p>
            </div>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_base64}" 
                    style="width: 90%; max-height: 450px; object-fit: contain; border: 1px solid #ddd; border-radius: 4px;"/>
            </div>
            <div style="margin-top: 8px;">
                <p><b>Interpretation:</b></p>
                <ul>
                    <li>Red/yellow areas: Highly influential regions for classification</li>
                    <li>Blue/green areas: Less important regions</li>
                </ul>
            </div>
        </div>
        """
        
        # Display the visualization in the prediction tab
        self.pred_result_text.setHtml(result_html)
    
    def visualize_rose_data(self):
        # For Rose dataset, create a bar chart showing class distribution
        if 'label_names' in self.metadata and self.metadata['label_names']:
            # Clear the figure
            self.data_canvas.fig.clear()
            ax = self.data_canvas.fig.add_subplot(111)
            
            # Count samples per class
            unique_classes, class_counts = np.unique(self.y_train, return_counts=True)
            
            # Get class names
            class_names = [self.metadata['label_names'][i] for i in unique_classes]
            
            # Create bar chart
            bars = ax.bar(class_names, class_counts, color='skyblue', width=0.6, edgecolor='navy', alpha=0.8)
            
            # Add count labels on top of each bar
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontsize=10)
            
            # Styling
            ax.set_title('Class Distribution in Rose Dataset', fontsize=14)
            ax.set_xlabel('Rose Classes', fontsize=12)
            ax.set_ylabel('Number of Samples', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Adjust to prevent label cutoff
            plt.xticks(rotation=45, ha='right')
            self.data_canvas.fig.tight_layout(pad=1.5)
            
            # Add information about dataset balance
            min_count = min(class_counts)
            max_count = max(class_counts)
            balance_ratio = min_count / max_count
            
            balance_text = f"Dataset Balance: "
            if balance_ratio > 0.9:
                balance_text += "Well balanced"
            elif balance_ratio > 0.5:
                balance_text += "Moderately balanced"
            else:
                balance_text += "Imbalanced"
            
            # Move the annotation to the right of the chart
            ax.annotate(balance_text, xy=(1.05, 0.5), xycoords='axes fraction', 
                        ha='left', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', 
                                                          fc='lightyellow', alpha=0.8))
        else:
            # If we don't have class names, display a message
            ax = self.data_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Image visualization not available", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    def display_dataset_info(self):
        info_text = f"Dataset: {self.metadata['dataset_type']}\n"
        info_text += f"Training samples: {len(self.X_train)}\n"
        info_text += f"Testing samples: {len(self.X_test)}\n"
        
        if self.metadata['dataset_type'] == 'Iris':
            info_text += f"Features: {', '.join(self.metadata['feature_names'])}\n"
        else:
            if 'image_size' in self.metadata:
                info_text += f"Image size: {self.metadata['image_size'][0]}x{self.metadata['image_size'][1]}\n"
        
        info_text += f"Number of classes: {len(np.unique(self.y_train))}\n"
        
        if 'label_names' in self.metadata and self.metadata['label_names'] is not None:
            # Convert to list if it's a numpy array
            label_names = (self.metadata['label_names'].tolist() 
                        if isinstance(self.metadata['label_names'], np.ndarray) 
                        else self.metadata['label_names'])
            info_text += f"Classes: {', '.join(str(label) for label in label_names)}\n"
        
        info_text += "\nData preparation steps:\n"
        info_text += "1. Loading dataset\n"
        info_text += "2. Extracting features and labels\n"
        info_text += "3. Splitting into training and testing sets\n"
        info_text += "4. Feature scaling\n"
        
        if self.metadata['dataset_type'] == 'Rose':
            info_text += "5. Image resizing and flattening\n"
            if len(self.X_train[0]) <= 100:
                info_text += "6. Dimensionality reduction (PCA)\n"
        
        self.data_info_text.setPlainText(info_text)
    
    def train_model(self):
        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "Error", "Please load and prepare dataset first.")
            return
        
        self.set_status("Training SVM model...")
        self.update_progress(0)
        
        # Start the model training worker
        self.model_training_worker = ModelTrainingWorker(self.X_train, self.y_train, optimization=False)
        self.model_training_worker.progress_update.connect(self.update_progress)
        self.model_training_worker.status_update.connect(self.set_status)
        self.model_training_worker.completed.connect(self.on_model_training_complete)
        self.model_training_worker.start()
        
        # Disable UI elements during training
        self.train_model_btn.setEnabled(False)
        self.optimize_model_btn.setEnabled(False)
        self.load_dataset_btn.setEnabled(False)
    
    def optimize_model(self):
        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "Error", "Please load and prepare dataset first.")
            return
        
        # Show information about optimization
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Model optimization will perform hyperparameter tuning using Grid Search.")
        msg.setInformativeText("This process may take several minutes. Do you want to continue?")
        msg.setWindowTitle("Model Optimization")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if msg.exec_() == QMessageBox.No:
            return
        
        self.set_status("Starting model optimization...")
        self.update_progress(0)
        
        # Start the model training worker with optimization
        self.model_training_worker = ModelTrainingWorker(self.X_train, self.y_train, optimization=True)
        self.model_training_worker.progress_update.connect(self.update_progress)
        self.model_training_worker.status_update.connect(self.set_status)
        self.model_training_worker.completed.connect(self.on_model_training_complete)
        self.model_training_worker.start()
        
        # Disable UI elements during training
        self.train_model_btn.setEnabled(False)
        self.optimize_model_btn.setEnabled(False)
        self.load_dataset_btn.setEnabled(False)
    
    def on_model_training_complete(self, model, model_info):
        # Store the model and info
        self.model = model
        self.model_info = model_info
        
        # Update UI
        self.set_status(f"Model training complete. {model_info['name']} ready for evaluation.")
        self.train_model_btn.setEnabled(True)
        self.optimize_model_btn.setEnabled(True)
        self.load_dataset_btn.setEnabled(True)
        self.evaluate_model_btn.setEnabled(True)
        self.save_model_btn.setEnabled(True)
        self.load_image_btn.setEnabled(True)
        
        # Display model information
        model_info_text = f"Model: {model_info['name']}\n\n"
        model_info_text += f"Parameters:\n"
        
        for param, value in model_info['params'].items():
            model_info_text += f"- {param}: {value}\n"
        
        if model_info['optimized']:
            model_info_text += "\nOptimization Results:\n"
            model_info_text += f"Best parameters: {model_info['best_params']}\n"
        
        self.data_info_text.setPlainText(model_info_text)
        
        # Switch to data visualization tab to show model info
        self.tab_widget.setCurrentIndex(0)
    
    def save_model(self):
        if self.model is None:
            QMessageBox.warning(self, "Error", "No model to save.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Pickle Files (*.pkl)")
        
        if filepath:
            try:
                # Save the model and metadata
                save_data = {
                    'model': self.model,
                    'model_info': self.model_info,
                    'metadata': self.metadata,
                    'feature_size': len(self.X_train[0])  # Store the expected feature size
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(save_data, f)
                
                self.set_status(f"Model saved to {os.path.basename(filepath)}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Pickle Files (*.pkl)")
        
        if filepath:
            try:
                with open(filepath, 'rb') as f:
                    save_data = pickle.load(f)
                
                # Load the model and metadata
                self.model = save_data['model']
                self.model_info = save_data['model_info']
                self.metadata = save_data['metadata']
                
                self.set_status(f"Model loaded from {os.path.basename(filepath)}")
                
                # Update UI
                self.evaluate_model_btn.setEnabled(True)
                self.save_model_btn.setEnabled(True)
                self.load_image_btn.setEnabled(True)
                
                # Display model information
                model_info_text = f"Loaded Model: {self.model_info['name']}\n\n"
                model_info_text += f"Dataset type: {self.metadata['dataset_type']}\n\n"
                model_info_text += f"Parameters:\n"
                
                for param, value in self.model_info['params'].items():
                    model_info_text += f"- {param}: {value}\n"
                
                if self.model_info['optimized']:
                    model_info_text += "\nOptimization Results:\n"
                    model_info_text += f"Best parameters: {self.model_info['best_params']}\n"
                
                self.data_info_text.setPlainText(model_info_text)
                
                # Switch to data visualization tab to show model info
                self.tab_widget.setCurrentIndex(0)
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load model: {str(e)}")

    
    def evaluate_model(self):
        if self.model is None:
            QMessageBox.warning(self, "Error", "Please train a model first.")
            return
        
        if self.X_test is None or self.y_test is None:
            QMessageBox.warning(self, "Error", "No test data available.")
            return
        
        self.set_status("Evaluating model performance...")
        self.update_progress(0)
        
        try:
            # Predict on test set
            self.update_progress(20)
            y_pred = self.model.predict(self.X_test)
            self.update_progress(40)
            
            # Ensure labels are properly formatted for metrics calculation
            y_test_eval = self.y_test
            y_pred_eval = y_pred
            
            # Calculate probabilities if available
            try:
                y_prob = self.model.predict_proba(self.X_test)
            except:
                y_prob = None
            
            self.update_progress(60)
            
            # Calculate metrics
            conf_matrix = confusion_matrix(y_test_eval, y_pred_eval)
            
            # Use zero_division parameter to handle zero divisions
            precision = precision_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
            recall = recall_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
            accuracy = accuracy_score(y_test_eval, y_pred_eval)
            
            self.update_progress(80)
            
            # Generate classification report
            if hasattr(self.metadata, 'label_names') and self.metadata['label_names'] is not None:
                target_names = self.metadata['label_names']
                report = classification_report(y_test_eval, y_pred_eval, 
                                            target_names=target_names, 
                                            zero_division=0)
            else:
                report = classification_report(y_test_eval, y_pred_eval, 
                                            zero_division=0)
            
            # Store evaluation results
            self.evaluation_results = {
                'y_pred': y_pred_eval,
                'y_prob': y_prob,
                'confusion_matrix': conf_matrix,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'report': report
            }
            
            self.update_progress(100)
            self.set_status("Model evaluation complete")
            
            # Visualize evaluation results
            self.visualize_evaluation_results()
            
            # Switch to evaluation tab
            self.tab_widget.setCurrentIndex(1)
            
        except Exception as e:
            import traceback
            traceback.print_exc()  # Print the full error traceback for debugging
            self.set_status(f"Error during model evaluation: {str(e)}")
            self.update_progress(0)
        
    def visualize_evaluation_results(self):
        if self.evaluation_results is None:
            return
        
        # Display metrics in text widget
        metrics_text = f"Model: {self.model_info['name']}\n\n"
        metrics_text += f"Precision: {self.evaluation_results['precision']:.4f}\n"
        metrics_text += f"Recall: {self.evaluation_results['recall']:.4f}\n"
        metrics_text += f"Accuracy: {self.evaluation_results['accuracy']:.4f}\n\n"
        metrics_text += "Classification Report:\n"
        metrics_text += self.evaluation_results['report']
        
        self.eval_metrics_text.setPlainText(metrics_text)
        
        # Plot confusion matrix
        self.conf_matrix_canvas.fig.clear()
        ax = self.conf_matrix_canvas.fig.add_subplot(111)
        
        # Get the confusion matrix
        cm = self.evaluation_results['confusion_matrix']
        
        # Create proper labels for the confusion matrix
        if self.metadata['label_names'] is not None and len(self.metadata['label_names']) > 0:
            labels = self.metadata['label_names']
        else:
            labels = [str(i) for i in range(len(cm))]
        
        # Plot with seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=ax, 
                    annot_kws={"size": 10}, cbar_kws={'shrink': 0.8})
        
        # Rotate x-axis labels for better fit
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_title('Confusion Matrix', fontsize=12)
        
        # Add more space at the bottom and right for labels
        self.conf_matrix_canvas.fig.subplots_adjust(bottom=0.15, right=0.9)
        self.conf_matrix_canvas.draw()
        
        # Plot precision-recall
        self.precision_recall_canvas.fig.clear()
        ax = self.precision_recall_canvas.fig.add_subplot(111)
        
        # Check if we have probability scores and handle multiclass case differently
        if self.evaluation_results['y_prob'] is not None:
            from sklearn.preprocessing import label_binarize
            
            # Get number of classes
            n_classes = len(np.unique(self.y_test))
            
            # For multiclass classification, we'll create a separate PR curve for each class
            if n_classes > 2:  # Multiclass case
                # Create binary labels for each class
                y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
                
                # Colors for different classes
                colors = ['red', 'green', 'cyan', 'blue', 'orange', 'purple']
                
                # Limit to max 3 classes for visibility
                class_display = min(n_classes, 3)
                
                # Plot PR curve for each class
                for i in range(class_display):
                    # Use OneVsRest approach - class i vs the rest
                    from sklearn.metrics import precision_recall_curve, average_precision_score
                    
                    # Get the probability scores for this class
                    y_score = self.evaluation_results['y_prob'][:, i]
                    
                    # Calculate precision-recall curve
                    precision, recall, _ = precision_recall_curve(
                        y_test_bin[:, i], y_score
                    )
                    
                    avg_precision = average_precision_score(y_test_bin[:, i], y_score)
                    
                    # Get class name
                    if self.metadata['label_names'] is not None and len(self.metadata['label_names']) > 0:
                        class_name = str(self.metadata['label_names'][i])
                        # Shorten long names
                        if len(class_name) > 10:
                            class_name = class_name[:8] + ".."
                    else:
                        class_name = f"Class {i}"
                    
                    # Plot the curve
                    ax.plot(
                        recall, precision, 
                        color=colors[i % len(colors)],
                        lw=2,
                        label=f"{class_name} (AP={avg_precision:.2f})"
                    )
            
            else:  # Binary case (unlikely for Rose dataset but keeping for completeness)
                from sklearn.metrics import precision_recall_curve, average_precision_score
                
                # For binary classification
                y_score = self.evaluation_results['y_prob'][:, 1]  # probability of positive class
                precision, recall, _ = precision_recall_curve(self.y_test, y_score)
                
                # Plot precision-recall curve
                ax.plot(recall, precision, lw=2, color='red',
                    label=f'AP={average_precision_score(self.y_test, y_score):.2f}')
                
        else:
            # If no probability scores available, just indicate that PR curve needs probabilities
            ax.text(0.5, 0.5, "Probability scores needed for PR curve",
                ha='center', va='center', fontsize=12)
        
        # Styling the graph
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title('Precision-Recall Curve', fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Add legend if we have curve data
        if self.evaluation_results['y_prob'] is not None:
            ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.8)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set blue background
        ax.set_facecolor('#f0f8ff')
        self.precision_recall_canvas.fig.patch.set_facecolor('#e6f0ff')
        
        # Adjust margins
        self.precision_recall_canvas.fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
        self.precision_recall_canvas.draw()
        
        # Create performance metrics line graph
        self.accuracy_canvas.fig.clear()
        ax = self.accuracy_canvas.fig.add_subplot(111)

        # Use actual metrics values
        accuracy = self.evaluation_results['accuracy']
        precision = self.evaluation_results['precision']
        recall = self.evaluation_results['recall']

        # Create x-axis categories
        categories = ['Training', 'Validation', 'Test']

        # Create metrics dictionary with values for visualization
        metrics = {
            'Accuracy': [accuracy, accuracy, accuracy],
            'Precision': [precision, precision, precision],
            'Recall': [recall, recall, recall]
        }

        # Plot each metric as a line
        colors = ['#2185d0', '#21ba45', '#f2711c']
        markers = ['o', 's', '^']

        for i, (metric_name, values) in enumerate(metrics.items()):
            ax.plot(categories, values, marker=markers[i], linestyle='-', 
                    linewidth=2, markersize=6, label=metric_name, color=colors[i])

        # Adjust y-axis to show small variations
        y_min = min(accuracy, precision, recall) * 0.98
        y_max = 1.01
        ax.set_ylim(y_min, y_max)

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Performance Metrics', fontsize=12)
        ax.set_xlabel('Dataset Split', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        
        # Format y-axis as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        ax.legend(loc='lower right', fontsize=8)
        
        # Set background color
        ax.set_facecolor('#f9f9f9')
        self.accuracy_canvas.fig.patch.set_facecolor('#f0f0f0')
        
        # Adjust margins
        self.accuracy_canvas.fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
        self.accuracy_canvas.draw()
        
    
    def load_new_image(self):
        if self.model is None:
            QMessageBox.warning(self, "Error", "Please train a model first.")
            return
        
        # For Iris dataset, we need to create a special dialog
        if self.metadata['dataset_type'] == 'Iris':
            # Create a dialog to enter feature values
            from PyQt5.QtWidgets import QDialog, QFormLayout, QDoubleSpinBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Enter Iris Features")
            layout = QFormLayout()
            
            feature_inputs = []
            
            # Add input fields for each feature
            for feature_name in self.metadata['feature_names']:
                spin_box = QDoubleSpinBox()
                spin_box.setRange(0, 10)
                spin_box.setDecimals(1)
                spin_box.setSingleStep(0.1)
                feature_inputs.append(spin_box)
                layout.addRow(feature_name, spin_box)
            
            # Add OK and Cancel buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addRow(button_box)
            
            dialog.setLayout(layout)
            
            if dialog.exec_():
                # Get the feature values
                feature_values = [input_field.value() for input_field in feature_inputs]
                
                # Create a fake "image" for display
                self.current_image = np.array(feature_values)
                self.current_image_path = None
                
                # Display the feature values
                feature_text = "Iris Features:\n\n"
                for name, value in zip(self.metadata['feature_names'], feature_values):
                    feature_text += f"{name}: {value}\n"
                
                self.pred_result_text.setPlainText(feature_text)
                
                # Create a visual representation for display
                self.image_label.clear()
                self.image_label.setText("Iris Sample\n(No image available)")
                
                # Enable verify button
                self.verify_image_btn.setEnabled(True)
                
                # Switch to prediction tab
                self.tab_widget.setCurrentIndex(2)
        else:
             # For image dataset
            filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
            
            if filepath:
                try:
                    # Load the original image for display
                    img_display = cv2.imread(filepath)
                    if img_display is None:
                        raise ValueError("Failed to load image")
                    
                    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                    
                    # Load and preprocess the image for classification
                    img = cv2.imread(filepath)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to the expected size for classification
                    image_size = self.metadata.get('image_size', (64, 64))
                    img_resized = cv2.resize(img, image_size)
                    
                    # Store flattened image for classification
                    self.current_image = img_resized.flatten()
                    self.current_image_path = filepath
                    
                    # Display the original image at a higher resolution
                    # We'll resize it to fit the label but maintain a minimum size
                    display_size = (256, 256)  # Use a larger display size
                    img_display_resized = cv2.resize(img_display, display_size, 
                                                    interpolation=cv2.INTER_LANCZOS4)  # Higher quality interpolation
                    
                    height, width, channel = img_display_resized.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(img_display_resized.data, width, height, 
                                bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    # Set the pixmap directly without additional scaling
                    self.image_label.setPixmap(pixmap)
                    
                    # Enable verify button
                    self.verify_image_btn.setEnabled(True)
                    
                    self.gradcam_btn.setEnabled(False)

                    # Clear previous results
                    self.pred_result_text.clear()
                    
                    # Switch to prediction tab
                    self.tab_widget.setCurrentIndex(2)
                    
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")
    
    def classify_image(self):
        import io
        import base64

        if self.model is None:
            QMessageBox.warning(self, "Error", "Please train a model first.")
            return

        if self.current_image is None:
            QMessageBox.warning(self, "Error", "Please load an image first.")
            return

        try:
            if self.metadata['dataset_type'] == 'Rose':
                img = cv2.imread(self.current_image_path)
                if img is None:
                    raise ValueError("Failed to load image for classification")

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_size = self.metadata.get('image_size', (64, 64))
                img_resized = cv2.resize(img, image_size)

                color_features = DataPreparationWorker.extract_color_features(img_resized)
                img_features = img_resized.flatten()

                if 'pca' in self.metadata and self.metadata['pca'] is not None:
                    pca = self.metadata['pca']
                    img_features = pca.transform([img_features])[0]

                color_weight = self.metadata.get('color_weight', 2.0)
                weighted_color_features = color_features * color_weight
                X = np.hstack(([img_features], [weighted_color_features]))

                if 'scaler' in self.metadata and self.metadata['scaler'] is not None:
                    X = self.metadata['scaler'].transform(X)

                dom_colors, percentages = self.extract_dominant_colors(img_resized, k=8)
                avg_color = np.mean(img_resized.reshape(-1, 3), axis=0)
                r_avg, g_avg, b_avg = avg_color
                hsv_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0] for color in dom_colors]

                yellow_detected = False
                pink_detected = False
                yellow_percentage = 0
                pink_percentage = 0
                red_percentage = 0
                color_names = []
                for i, hsv in enumerate(hsv_colors):
                    h, s, v = hsv
                    if s < 50 and v > 200:
                        color_names.append("White")
                    elif s < 60 and v < 150:
                        color_names.append("Gray/Brown")
                    elif 30 <= h <= 70 and s > 75:
                        color_names.append("Yellow")
                        yellow_detected = True
                        yellow_percentage += percentages[i]
                    elif h < 10 or h > 170:
                        color_names.append("Red")
                        red_percentage += percentages[i]
                    elif 10 <= h < 30:
                        color_names.append("Orange-Red")
                    elif 70 <= h < 100:
                        color_names.append("Yellow-Green")
                    elif 100 <= h < 140:
                        color_names.append("Green")
                    elif 140 <= h < 170 and s > 50:
                        color_names.append("Pink")
                        pink_detected = True
                        pink_percentage += percentages[i]
                    else:
                        color_names.append("Other")

                y_pred = self.model.predict(X)
                try:
                    probabilities = self.model.predict_proba(X)[0]
                    has_probabilities = True
                except:
                    probabilities = None
                    has_probabilities = False

                pred_class_idx = y_pred[0]
                if 'label_names' in self.metadata and len(self.metadata['label_names']) > 0:
                    pred_class_name = self.metadata['label_names'][pred_class_idx]
                    class_labels = self.metadata['label_names']
                else:
                    pred_class_name = f"Class {pred_class_idx}"
                    class_labels = [f"Class {i}" for i in range(len(probabilities))] if probabilities is not None else []

                original_class = pred_class_name
                color_override = False

                # --- Improved color override logic: only after optimization ---
                if self.model_info and self.model_info.get('optimized', False):
                    # Use mean HSV for robust color detection
                    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
                    mean_hue = np.mean(hsv_img[..., 0])
                    mean_sat = np.mean(hsv_img[..., 1])
                    mean_val = np.mean(hsv_img[..., 2])

                    # Pink: hue 140-170, high saturation, high value
                    is_pink = (140 <= mean_hue <= 170) and (mean_sat > 60) and (mean_val > 100)
                    # Yellow: hue 30-70, high saturation, high value
                    is_yellow = (30 <= mean_hue <= 70) and (mean_sat > 60) and (mean_val > 100)

                    # Require a higher percentage for override
                    if pink_detected and pink_percentage > 0.35 and is_pink and pred_class_name != "pink_rose":
                        for i, class_name in enumerate(class_labels):
                            if class_name == "pink_rose":
                                pred_class_idx = i
                                pred_class_name = "pink_rose"
                                color_override = True
                                break
                    elif yellow_detected and yellow_percentage > 0.35 and is_yellow and pred_class_name != "yellow_rose":
                        for i, class_name in enumerate(class_labels):
                            if class_name == "yellow_rose":
                                pred_class_idx = i
                                pred_class_name = "yellow_rose"
                                color_override = True
                                break

                # --- Modern HTML Output ---
                result_html = f"""
                <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #222;">
                <div style="margin-bottom: 10px;">
                    <span style="font-size: 16px; font-weight: bold; color: #2185d0;">Classification Result</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <b>Predicted Class:</b>
                    <span style="font-size: 15px; font-weight: bold; color: #21ba45;">{pred_class_name}</span>
                """
                if color_override:
                    result_html += f"""<span style="color: #f2711c; font-size: 12px;"> (adjusted from {original_class} based on color analysis)</span>
                    <div style="color: #f2711c; font-size: 12px;">Note: Color correction is only applied with optimized model.</div>
                    """
                result_html += "</div>"

                # Probabilities as horizontal bar graph (matplotlib)
                if has_probabilities and probabilities is not None:
                    fig, ax = plt.subplots(figsize=(4, 0.5 + 0.4 * len(probabilities)))
                    y_pos = np.arange(len(probabilities))
                    bar_colors = ['#21ba45' if i == np.argmax(probabilities) else '#2185d0' for i in range(len(probabilities))]
                    ax.barh(y_pos, probabilities * 100, color=bar_colors, edgecolor='black')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(class_labels)
                    ax.invert_yaxis()
                    ax.set_xlabel('Probability (%)')
                    ax.set_xlim(0, 100)
                    for i, v in enumerate(probabilities):
                        ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va='center', fontsize=10)
                    ax.set_title("Class Probabilities", fontsize=11)
                    plt.tight_layout()

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    result_html += f"""
                    <div style="margin-bottom: 10px;">
                    <img src="data:image/png;base64,{img_base64}" style="max-width: 100%;"/>
                    </div>
                    """

                # Color analysis section
                result_html += """
                <div style="margin-bottom: 10px;">
                <b>Color Analysis:</b>
                <div style="margin-top: 5px;">
                """
                for i, (color_name, percentage) in enumerate(zip(color_names, percentages)):
                    result_html += f"Color {i+1}: <span style='color:#2185d0'>{color_name}</span> ({percentage*100:.1f}%)<br>"
                result_html += "</div></div>"

                # Average RGB
                result_html += f"""
                <div style="margin-bottom: 10px;">
                <b>Average RGB:</b> R={r_avg:.1f}, G={g_avg:.1f}, B={b_avg:.1f}
                </div>
                """

                # Summary percentages
                if pink_percentage > 0:
                    result_html += f"<div>Pink color: <b>{pink_percentage*100:.1f}%</b></div>"
                if yellow_percentage > 0:
                    result_html += f"<div>Yellow color: <b>{yellow_percentage*100:.1f}%</b></div>"
                if red_percentage > 0:
                    result_html += f"<div>Red color: <b>{red_percentage*100:.1f}%</b></div>"

                # Notes for potential misclassification
                if not self.model_info.get('optimized', False):
                    if pink_detected and pink_percentage > 0.25 and pred_class_name != "pink_rose":
                        result_html += f"""
                        <div style="color: #f2711c; margin-top: 8px;">
                        <b>Potential issue:</b> Pink colors detected but classified as {pred_class_name}.<br>
                        Optimized model would apply color-based correction for this case.
                        </div>
                        """
                    if yellow_detected and yellow_percentage > 0.25 and pred_class_name != "yellow_rose":
                        result_html += f"""
                        <div style="color: #f2711c; margin-top: 8px;">
                        <b>Potential issue:</b> Yellow colors detected but classified as {pred_class_name}.<br>
                        Optimized model would apply color-based correction for this case.
                        </div>
                        """

                result_html += "</div>"
                self.pred_result_text.setHtml(result_html)
                
                # Enable Grad-CAM button after successful classification for Rose dataset
                self.gradcam_btn.setEnabled(True)

            else:
                # For Iris dataset (unchanged, but with horizontal bar graph)
                X = np.array([self.current_image])
                if 'scaler' in self.metadata and self.metadata['scaler'] is not None:
                    X = self.metadata['scaler'].transform(X)

                y_pred = self.model.predict(X)
                try:
                    probabilities = self.model.predict_proba(X)[0]
                    has_probabilities = True
                except:
                    probabilities = None
                    has_probabilities = False

                pred_class_idx = y_pred[0]
                if 'label_names' in self.metadata and len(self.metadata['label_names']) > 0:
                    pred_class_name = self.metadata['label_names'][pred_class_idx]
                    class_labels = self.metadata['label_names']
                else:
                    pred_class_name = f"Class {pred_class_idx}"
                    class_labels = [f"Class {i}" for i in range(len(probabilities))] if probabilities is not None else []

                result_html = f"""
                <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #222;">
                <div style="margin-bottom: 10px;">
                    <span style="font-size: 16px; font-weight: bold; color: #2185d0;">Classification Result</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <b>Predicted Class:</b>
                    <span style="font-size: 15px; font-weight: bold; color: #21ba45;">{pred_class_name}</span>
                </div>
                """

                if has_probabilities and probabilities is not None:
                    fig, ax = plt.subplots(figsize=(4, 0.5 + 0.4 * len(probabilities)))
                    y_pos = np.arange(len(probabilities))
                    bar_colors = ['#21ba45' if i == np.argmax(probabilities) else '#2185d0' for i in range(len(probabilities))]
                    ax.barh(y_pos, probabilities * 100, color=bar_colors, edgecolor='black')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(class_labels)
                    ax.invert_yaxis()
                    ax.set_xlabel('Probability (%)')
                    ax.set_xlim(0, 100)
                    for i, v in enumerate(probabilities):
                        ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va='center', fontsize=10)
                    ax.set_title("Class Probabilities", fontsize=11)
                    plt.tight_layout()

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    result_html += f"""
                    <div style="margin-bottom: 10px;">
                    <img src="data:image/png;base64,{img_base64}" style="max-width: 100%;"/>
                    </div>
                    """

                result_html += "</div>"
                self.pred_result_text.setHtml(result_html)
                
                # Make sure Grad-CAM is disabled for Iris
                self.gradcam_btn.setEnabled(False)

            self.set_status(f"Image classified as: {pred_class_name}")

        except Exception as e:
            self.set_status(f"Error during classification: {str(e)}")
            import traceback
            traceback.print_exc()
            # Disable Grad-CAM button in case of errors
            self.gradcam_btn.setEnabled(False)
            
    def show_help(self):
        help_text = """
<h3>Image Classification System Help</h3>
<p>This application allows you to train and evaluate SVM-based image classification models on the Iris or Rose datasets.</p>

<h4>Workflow:</h4>
<ol>
    <li><b>Data Preparation:</b> Select a dataset type and load the dataset.</li>
    <li><b>Model Training:</b> Train a basic SVM model or use hyperparameter optimization.</li>
    <li><b>Model Evaluation:</b> Evaluate the model's performance on test data.</li>
    <li><b>Prediction:</b> Load new images/samples and classify them using the trained model.</li>
</ol>

<h4>Dataset Information:</h4>
<ul>
    <li><b>Iris Dataset:</b> CSV file with numerical features.</li>
    <li><b>Rose Dataset:</b> Directory with rose images organized in class folders.</li>
</ul>

<h4>Tips:</h4>
<ul>
    <li>The model optimization option will perform grid search to find optimal hyperparameters.</li>
    <li>You can save trained models for later use.</li>
    <li>For Iris data, you'll enter feature values manually instead of loading an image.</li>
</ul>
"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Help")
        msg.setText(help_text)
        msg.exec_()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def set_status(self, message):
        self.status_label.setText(message)


def main():
    app = QApplication(sys.argv)
    window = ImageClassificationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()