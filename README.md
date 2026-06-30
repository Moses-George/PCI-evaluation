# Pavement Condition Index Evaluation using Deep Learning (Python)

## YoloV8 | UNET

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Moses-George/PCI-evaluation.git
cd PCI-evaluation
```

## 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### Example requirements.txt

```
torch
torchvision
scipy
scikit-learn
opencv-python
tqdm
pandas
matplotlib
scikit-image
```

---

## References

Tello-Cifuentes, L., Thomson, P., Marulanda, J., Sandoval, E., & Bassier, M. (2026).
Deep learning-based detection and evaluation of pavement surface damage.
Results in Engineering, 29, 108946.
https://doi.org/10.1016/j.rineng.2025.108946

ASTM D6433-07; Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys. ASTM: West Conshohocken,
PA, USA, 2007.

<!-- Karan Sharma, Barun Das, Dr. Ravi A Patel, and Dr. Kranthi Kumar Kuna. (2025). UAV-SEG-2025: Segmented Pavement Distress Dataset [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/10883009 -->

## Dataset Citation

This project uses the **Road Potholes and Cracks Dataset** by **smark**, nd **road damage dét Dataset** by **baka** available on Roboflow Universe. If you use this dataset, please cite it as follows:

```bibtex
@misc{road-potholes-and-cracks-jmxtp_dataset,
  title        = {Road Potholes and Cracks Dataset},
  author       = {smark},
  year         = {2024},
  month        = {aug},
  publisher    = {Roboflow},
  journal      = {Roboflow Universe},
  howpublished = {\url{https://universe.roboflow.com/smark-z7pr6/road-potholes-and-cracks-jmxtp}},
  url          = {https://universe.roboflow.com/smark-z7pr6/road-potholes-and-cracks-jmxtp},
  note         = {Accessed: 2026-06-17}
}

@misc{ road-damage-det_dataset,
  title = { road damage dét Dataset },
  type = { Open Source Dataset },
  author = { baka },
  howpublished = { \url{ https://universe.roboflow.com/baka-1ravj/road-damage-det } },
  url = { https://universe.roboflow.com/baka-1ravj/road-damage-det },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2023 },
  month = { may },
  note = { visited on 2026-06-20 },
}
}
```

### Digitizer Tool
```bibtex
@misc{Rohatgi2024WebPlotDigitizer,
  author = {Ankit Rohatgi},
  title = {WebPlotDigitizer},
  year = {2024},
  version = {5.3},
  howpublished = {\url{https://automeris.io/WebPlotDigitizer/}}
}
```
