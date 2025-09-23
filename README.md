# garvis

## Set the environment
```bash
python3 -m venv venv
source venv/bin/activate
python -r pip install < requirements.txt
export PYTHONPATH=.
```

## Generate 2D scenarios


```bash
usage: scenario2d.py [-h] [--output OUTPUT] [--n_experiments N_EXPERIMENTS] [--area_size AREA_SIZE] [--speed_min SPEED_MIN]
                     [--speed_max SPEED_MAX] [--input INPUT] [--av AV] [--print]
                     {generate,visualize}
```

## Visualize 2D scenarios

To print images of 3 scenarios in directory data:

```bash
python3 scenario/scenario2d.py visualize --input data/traiettorie_aerei.csv --print --av 6
```
the charts will be saved data/2dscenario_{0..2}.png images.

## Run genetic algorithm
All parameters of the algorithm are set in scenario/conf.py

```bash
python3 ga/disiero.py
```

The result is a file named data/pareto_scenario_{i}.json for each scenario.

## GA result visualization

Read from data/pareto_scenario{i}.json and generate  images of the pareto and of paths.