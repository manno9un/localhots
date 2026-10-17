# localhots

A wildfire occurrence-risk ranking model and interactive historical replay, built for the IBM Z Datathon 2026 and executed on IBM LinuxONE.

**Live website:** [http://148.100.109.228/](http://148.100.109.228/)

localhots ranks 24 half-degree areas in Northern California for recorded wildfire discoveries on a target day using weather through the preceding day and historical fire/climate statistics. The website replays 2022–2024, displays the model scores on a map, and compares them with recorded discoveries. Areas without model data have a transparent heat overlay.

## Repository contents

```text
.
├── README.md
├── SERVER_FILE_MANIFEST.json
├── wildfire-site/
│   ├── app/
│   │   ├── server.py
│   │   └── static/
│   │       ├── index.html
│   │       ├── style.css
│   │       └── app.js
│   └── deploy/
│       └── run.sh
└── wildfire-outlook/
    ├── model/
    │   ├── model.joblib
    │   ├── predict.py
    │   ├── requirements.txt
    │   └── manifest.json
    └── datasets/
        ├── model_inputs.csv.gz
        ├── trial_inputs_2012_2021.csv.gz
        ├── included_fires.csv
        ├── manifest.json
        └── raw/
            ├── original/
            ├── gridmet/
            ├── usgs_batch.json
            └── usgs_terrain.json
```

The website, inference source, retained model, and datasets are directly browsable in this repository. The original server export inventory is adjusted to describe the retained payload files; generated Python bytecode is omitted. The prior server cleanup removed training/experiment source, so this snapshot supports inference and website deployment rather than a complete retraining workflow.

## Model and data

The retained V2 model uses scikit-learn `HistGradientBoostingClassifier` with 120 boosting iterations and 27 features. Its score blends 75% tree-model output with 25% smoothed historical cell/month fire frequency. Historical state ends on 2021-12-31; tree fitting used 2013–2021.

Inputs include recent temperature, humidity, wind, rainfall, dry-day duration, seasonality, calendar categories, historical fire frequency, and historical weather anomalies. Target-day weather and target outcome columns are excluded from model inputs.

The final package contains original USFS fire records and ERA5 weather responses, the precomputed feature panel, and the additional gridMET/USGS datasets evaluated during development. The retained model does not consume the additional gridMET/elevation features.

This is a retrospective research demo. The weather inputs are reanalysis, raw scores are not independently calibrated probabilities, and the 2022–2024 benchmark was reused during development. Live current-date forecasting is not implemented.

## Technology and IBM LinuxONE

- LinuxONE host: Linux s390x, three virtual CPUs and approximately 8 GB RAM.
- Python backend: standard-library HTTP server plus pandas, NumPy, scikit-learn and joblib.
- Scientific data support: SciPy.
- Frontend: HTML, CSS, JavaScript, Leaflet and Chart.js.
- Deployment: Docker using the pinned IBM-provided s390x JupyterLab image.

Exact model dependency versions and image digest are recorded in [model/manifest.json](wildfire-outlook/model/manifest.json). The application and model directories are mounted read-only into the deployed container.

## Deploy the website on LinuxONE

The supplied deployment script expects the two project directories at these paths:

```text
/home/linux1/wildfire-site
/home/linux1/wildfire-outlook
```

Copy them from this checkout to your LinuxONE host, using your own SSH key path:

```bash
scp -i /path/to/Datathon.pem -r wildfire-site wildfire-outlook linux1@YOUR_HOST:/home/linux1/
ssh -i /path/to/Datathon.pem linux1@YOUR_HOST
```

Then, on LinuxONE:

```bash
bash /home/linux1/wildfire-site/deploy/run.sh
curl http://127.0.0.1:8080/api/health
```

Docker must be installed and accessible through sudo. The script replaces the `wildfire-outlook` container, applies limits of three CPUs and 2 GB RAM, publishes the website on port 80 and loopback port 8080, and waits for the model server to become healthy. The pinned image targets s390x and is not assumed to run natively on an x86 laptop. Browser fonts, map tiles, Leaflet and Chart.js require network access.

API endpoints:

- `/api/health`: process health and architecture.
- `/api/overview`: model, data and evaluation information.
- `/api/day?date=2024-07-03`: scores and replay information for a selected date.

## Run model inference without the website

On the LinuxONE host, after placing the project at the paths above:

```bash
sudo docker run --rm --cpus 3 --memory 2g \
  -e OMP_NUM_THREADS=3 -e OPENBLAS_NUM_THREADS=3 \
  -v /home/linux1/wildfire-outlook:/work:ro -w /work \
  --entrypoint /opt/venv/bin/python \
  registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x@sha256:d2da8a7b0c88c3b468e29014b2ada9ef8274f98c976ec456904c5e2bae5aa54c \
  -B model/predict.py --date 2024-07-03
```

The CLI returns JSON scores and ranks for 24 cells. Add `--cell r2c2` to select a cell while retaining its rank among the supplied date's rows. Bundled replay inputs support dates in 2022–2024 for this fitted model. Dates at or before the training cutoff are rejected; current dates need newly prepared compatible input features.

## Data sources and attribution

- [US Forest Service FPA FOD, seventh edition](https://doi.org/10.2737/RDS-2013-0009.7): recorded wildfire discoveries.
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) / Copernicus ERA5: historical weather; retain Open-Meteo/Copernicus attribution under the terms recorded in the data manifest.
- [gridMET](https://www.climatologylab.org/gridmet.html): modeled fuel moisture and fire-danger indices evaluated in the final additional-data trial.
- [USGS 3DEP](https://www.usgs.gov/3d-elevation-program): centroid elevation evaluated in that trial.
- [OpenStreetMap](https://www.openstreetmap.org/copyright): map data and tiles with attribution.
- [Leaflet](https://github.com/Leaflet/Leaflet), [Chart.js](https://github.com/chartjs/Chart.js), and [Inter](https://github.com/rsms/inter): frontend libraries and font.

Source URLs and dataset payload hashes are in [datasets/manifest.json](wildfire-outlook/datasets/manifest.json). Third-party data and software retain their own licenses and attribution requirements.

## Artifact integrity

The retained model's SHA-256 is:

```text
f8160573882b6029b260e1c93be3f618dd3663f5690110c347bbcf782ec8e8b2
```

Load only the trusted supplied joblib artifact. The model and dataset manifests record the original payload hashes. Git attributes preserve those payload bytes without newline conversion. `SERVER_FILE_MANIFEST.json` records hashes of the copied server payloads included in this repository.
