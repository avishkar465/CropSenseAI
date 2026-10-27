# CropSense AI

CropSense AI is a locally run Flask application for recording crop-leaf image screenings. It includes accounts, an owner-scoped prediction history, a disease guide, MobileNetV2 training/evaluation scripts, and Grad-CAM support for compatible trained models.

**Important:** no dataset or trained model is included in this project. Until you train and evaluate a model, the app will not invent a diagnosis: uploads show a clear missing-model message. No accuracy or training result is claimed by this repository.

## Requirements

- Windows 11 and VS Code
- Python 3.11 (the workspace virtual environment currently uses Python 3.11.9)
- MySQL Server 8.0+ and MySQL Workbench
- Git, for obtaining the optional PlantVillage dataset
- Internet access for package installation and the remote Bootstrap/font/chart assets

TensorFlow can use substantial disk space and memory. CPU training is supported but may take a long time. A supported GPU setup is optional.

## 1. Create The Environment

Open the project folder in VS Code and open a PowerShell terminal. From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, allow it for this terminal session only, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

In VS Code, select `venv\Scripts\python.exe` with **Python: Select Interpreter**. Dependencies are installed into this project environment, not globally.

## 2. Configure MySQL

Install and start MySQL Server. In MySQL Workbench, connect to your local server and open a SQL tab. To create an application account with access limited to CropSense, run the following as a MySQL administrator. Replace the example password with a unique local password before running it:

```sql
CREATE USER 'cropsense_app'@'localhost' IDENTIFIED BY 'replace-with-a-unique-password';
GRANT CREATE ON *.* TO 'cropsense_app'@'localhost';
```

The temporary global `CREATE` privilege lets `init_db.py` create the configured database and its tables. After initialization, reduce this account to application-only data permissions:

```sql
REVOKE CREATE ON *.* FROM 'cropsense_app'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON cropsense_ai.* TO 'cropsense_app'@'localhost';
```

If your administrator creates the database and tables separately, the app account needs only the second grant; the schema is in `schema.sql`. Do not use a MySQL root account in `.env`.

Copy the environment example and edit it in VS Code:

```powershell
Copy-Item .env.example .env
```

Set `SECRET_KEY` to a long, private random value, then set `DB_USER` and `DB_PASSWORD` to the local MySQL account above. Keep `.env` private; it is ignored by Git. `COOKIE_SECURE=0` is suitable for local HTTP; enable it only when serving the app over HTTPS.

Initialize the database:

```powershell
python init_db.py
```

The initializer creates `cropsense_ai` (if permitted) and the `users` and `predictions` tables. If it reports access denied, ask the MySQL administrator to create the database or temporarily grant the privileges described above.

## 3. Prepare The Dataset

PlantVillage is an optional starting dataset. Read its official repository and usage terms: [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset). The upstream repository is maintained by its authors; use its current instructions rather than relying on a third-party repackaging.

One possible Windows workflow, after reviewing the upstream terms, is:

```powershell
git clone https://github.com/spMohanty/PlantVillage-Dataset.git data\PlantVillage-source
New-Item -ItemType Directory -Force data\PlantVillage | Out-Null
Copy-Item data\PlantVillage-source\raw\color\* data\PlantVillage\ -Recurse
```

The resulting directory must contain one directory per class, with images beneath each class folder, for example `data\PlantVillage\Tomato___Early_blight\...`. If the upstream folder layout changes, select its color-image class folders and arrange them this way. The training script makes separate, seeded per-class train, validation, and test file splits; it does not reuse a training image in validation or test.

## 4. Train And Evaluate

Training downloads ImageNet MobileNetV2 weights by default, then trains a crop-specific classification head and fine-tunes the final backbone layers. A held-out test split is used for reported metrics.

```powershell
python train_model.py
```

Optional settings:

```powershell
python train_model.py --data-dir data\PlantVillage --epochs 12 --fine-tune-epochs 5 --batch-size 32
```

To train without downloading ImageNet initialization weights (usually slower to converge):

```powershell
python train_model.py --weights none
```

The script saves `model\crop_disease_model.keras`, `model\class_names.json`, training history, test metrics, a confusion matrix image, and a best-checkpoint file. These generated model/data artifacts are ignored by Git. Keep the saved class labels paired with the exact model.

Run a separate held-out evaluation or verify loading/inference with a sample image:

```powershell
python evaluate_model.py
python verify_model.py
python verify_model.py --image path\to\leaf.jpg
```

Training and evaluation can be lengthy. Read `model\evaluation.json` and consider per-class metrics and the confusion matrix; a single aggregate score is not enough to establish reliable field performance.

## 5. Run The Web App

With MySQL configured and initialized, and a trained model installed, start Flask:

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Create an account, upload a JPG/PNG leaf image (maximum 8 MB), review the top three model classes and confidence, then open the saved result from History. The uploader and authenticated pages can be explored without a trained model; the prediction route will explicitly report that inference is unavailable.

## 6. Tests

Run the tests without MySQL or a trained model:

```powershell
python -m unittest discover -s tests -v
```

The suite uses a fake database and mock predictor to test authentication, CSRF, protected routes, uploads, prediction persistence, user-specific history, database configuration, and model/Grad-CAM error handling. It does not test real MySQL permissions, train the model, or establish real-world model accuracy.

## Troubleshooting

- **`Access denied` from MySQL:** verify MySQL Server is running and `.env` host, port, user, and password match. Run the one-time database creation grant as an administrator, then rerun `python init_db.py`.
- **`Unknown database` or missing tables:** run `python init_db.py` after checking `.env` and the user's initialization privileges.
- **Model unavailable:** add the prepared class-folder dataset and run `python train_model.py`. Both `model\crop_disease_model.keras` and `model\class_names.json` must exist and match.
- **Invalid image / upload too large:** use an actual JPG/JPEG/PNG image under 8 MB. The server checks decoded image content, not only the filename.
- **PowerShell activation is blocked:** use the process-only execution-policy command in the environment section; do not change machine-wide policy for this project.
- **TensorFlow install or memory problems:** confirm the VS Code interpreter is Python 3.11, use a 64-bit runtime, close memory-heavy programs, lower `--batch-size`, or install a compatible TensorFlow/Python combination. The app's authentication and guide do not need model training.
- **Grad-CAM unavailable:** the saved model must expose a connected convolutional layer. Results still display if explainability is unsupported; the app does not substitute a fake overlay.
- **Charts, fonts, or landing photo do not load offline:** those visual assets use CDN/remote URLs; local authentication, routing, and stored history remain local.

## Limitations And Responsible Use

- The output is a model estimate for supported dataset classes, not a diagnosis. Image quality, cultivar, growth stage, geography, lighting, and out-of-distribution images can produce misleading results.
- Confidence is the model's class score, not a calibrated probability of disease. Results under 60% are explicitly marked uncertain; scores above that threshold still need human review.
- Grad-CAM indicates image regions that influenced a model output. It does not prove a highlighted region is diseased or establish causation.
- The disease guide is general educational information. It does not prescribe pesticides, dosages, or chemical treatments. Confirm concerns and management decisions with a local agricultural extension service or qualified professional.
- PlantVillage images are controlled dataset images and do not represent every real farm condition. Evaluate on independent, locally representative images before any consequential use.
- Image files are stored in `instance\uploads` and heatmaps in `instance\heatmaps`; prediction records store relative paths. Back up or delete these files according to your privacy needs. The app does not include account deletion or automated data-retention management.
- The local Flask development server is not a production deployment configuration.