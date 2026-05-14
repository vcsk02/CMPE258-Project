.PHONY: setup test train train-quick evaluate app ablations sweeps manifest clean

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

test:
	PYTHONPATH=src pytest -q

train:
	python -m deep_har.train --model cnn_lstm --epochs 40 --batch-size 64

train-quick:
	python -m deep_har.train --model cnn --epochs 2 --quick-run

evaluate:
	python -m deep_har.evaluate --model-path outputs/checkpoints/cnn_lstm_best.keras --normalizer-path outputs/preprocessing/normalizer.npz

app:
	python app.py

ablations:
	python -m deep_har.ablations --epochs 15 --models mlp cnn cnn_lstm tcn attention sensor_transformer

sweeps:
	python -m deep_har.sweeps --epochs 10 --models cnn cnn_lstm sensor_transformer

manifest:
	python scripts/generate_artifact_manifest.py

clean:
	rm -rf outputs/checkpoints outputs/results outputs/plots outputs/tensorboard mlruns
