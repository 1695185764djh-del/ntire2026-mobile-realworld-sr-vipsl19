# Team 19 (VIPSL) Code Package for NTIRE 2026 Mobile Real-World SR
Official reproducibility package for our NTIRE 2026 Mobile Real-World Image Super-Resolution submission (Team VIPSL #19).
## Repository Contents
- `plksr/` — model/network code
- `scripts/` — inference helper scripts
- `test.py` — inference entry
- `eval.py` — quick output format check
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT license
> Note: Pretrained weights are hosted in **GitHub Releases** (not stored in repo tree due to file size limits).
## Environment
Tested with:
- Python 3.10
- PyTorch (CUDA)
Install dependencies:
```bash
pip install -r requirements.txt
Pretrained Weights (Release Asset)
Release page:
- https://github.com/1695185764djh-del/ntire2026-mobile-realworld-sr-vipsl19/releases/tag/ntire2026-final
Required checkpoint:
- net_g_1000.pth
Download and place into expected path:
mkdir -p model_zoo/19_PLKSRRep_IQAv2Short
wget -O model_zoo/19_PLKSRRep_IQAv2Short/net_g_1000.pth \
  "https://github.com/1695185764djh-del/ntire2026-mobile-realworld-sr-vipsl19/releases/download/ntire2026-final/net_g_1000.pth"
Run Inference
python test.py --input /path/to/test_LR --output /path/to/sr \
  --model_path model_zoo/19_PLKSRRep_IQAv2Short/net_g_1000.pth
Optional flags:
python test.py --input /path/to/test_LR --output /path/to/sr \
  --model_path model_zoo/19_PLKSRRep_IQAv2Short/net_g_1000.pth \
  --gpu 0 --fp16 0 --prepad 16
Quick Output Check
python eval.py --input /path/to/test_LR --output /path/to/sr --scale 4
Expected:
- Output PNG filenames match input filenames
- Output resolution is exactly 4x input resolution
Release Assets
The ntire2026-final release includes:
- net_g_1000.pth (pretrained checkpoint)
- final_candidate_iqav2short_ckpt1000_test_fp32_20260317_113426.zip (final FP32 output pack)
- MobileSR_factsheet.pdf
- MobileSR_factsheet_source.zip
Notes
- Final scoring outputs are generated in FP32.
- Extra data used: Yes.
License
This project is licensed under the MIT License. See LICENSE (./LICENSE).
