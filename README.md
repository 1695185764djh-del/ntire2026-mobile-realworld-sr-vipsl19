# Team 19 (VIPSL) code package for NTIRE 2026 Mobile Real-World SR

This package follows the requested layout:

- `models/19_PLKSRRep_IQAv2Short/`
- `model_zoo/19_PLKSRRep_IQAv2Short/`
- `test.py`
- `eval.py`

## Install

```bash
pip install -r requirements.txt
```

## Run inference

```bash
python test.py --input /path/to/test_LR --output /path/to/sr
```

Optional:

```bash
python test.py --input /path/to/test_LR --output /path/to/sr --model_path model_zoo/19_PLKSRRep_IQAv2Short/net_g_1000.pth --gpu 0 --fp16 0 --prepad 16
```

## Quick output check

```bash
python eval.py --input /path/to/test_LR --output /path/to/sr --scale 4
```

## Weight file used

- `model_zoo/19_PLKSRRep_IQAv2Short/net_g_1000.pth`
This project is licensed under the MIT License.
