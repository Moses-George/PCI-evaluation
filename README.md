# 💧 Hydraulic Flow Balancing Using Hardy Cross Method (Python)

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)
![Status](https://img.shields.io/badge/Status-Published-success.svg)
![Validation](https://img.shields.io/badge/R²-0.999-brightgreen.svg)

---

## 📄 Paper Under Review

This repository contains the computational implementation accompanying the paper under review:

### “A MODULAR PYTHON-BASED FRAMEWORK FOR AUTOMATED HYDRAULIC FLOW BALANCING IN LOOPED PIPE NETWORKS USING THE HARDY CROSS METHOD”

The research automates the classical Hardy Cross iterative method using Python, supporting both Hazen–Williams and Darcy–Weisbach head-loss equations.

---

## 🚀 Project Overview

Manual application of the Hardy Cross method becomes:

- Time-consuming  
- Error-prone  
- Inefficient for multi-loop systems  

This project provides a validated Python-based solution that:

- Automates loop corrections
- Dynamically computes friction factor (Swamee–Jain equation)
- Handles shared pipes in multi-loop systems
- Accepts structured Excel inputs
- Produces statistically validated outputs

---

## 🧠 Features

✅ Hardy Cross iterative loop balancing  
✅ Hazen–Williams head-loss equation  
✅ Darcy–Weisbach head-loss equation  
✅ Automatic friction factor calculation  
✅ Multi-loop sparse matrix handling  
✅ Excel-based input system  
✅ Statistical validation metrics (R², RMSE, MBE, MAE)  
✅ Benchmark-tested (4 hydraulic networks)  

---

## 📊 Validation Performance

| Metric | Value |
|--------|--------|
| R²     | 0.999  |
| RMSE   | ≤ 0.001 |
| MBE    | 0.000 |
| MAE    | ≤ 0.001 |

The results demonstrate near-perfect agreement with benchmark solutions.

---

# 📁 Project Structure

```
hardy-cross-python/
│
├── data/
│   ├── network1.xlsx
│   ├── network2.xlsx
│   ├── network3.xlsx
│   └── network4.xlsx
│
├── src/
│   ├── main.py
│   ├── hardy_cross.py
│   ├── headloss.py
│   ├── friction_factor.py
│   ├── correction.py
│   └── utils.py
│
├── results/
│   ├── network1_results.xlsx
│   ├── network2_results.xlsx
│   └── plots/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Moses-George/Hardy-Cross.git
cd Hardy-Cross
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
numpy
pandas
openpyxl
matplotlib
```

---

# 📝 Input Data Format

The program accepts Excel files with structured columns such as:

Please copy exercise file from /data/... to understand pipe characteristics structure properly

| Pipe (Pipe Id e.g AB) | L (Length (m)) | Q (Initial Flow (m³/s)) | D (Diameter (m)) | e (Roughness) |  
|-----------------------|------|----|------------|--------------|-----------|---------------------|----------|

- Flow directions must follow assumed orientation.
- Initial flows must satisfy continuity at nodes.
- Loop IDs define balancing groups.

---

# ▶️ Usage Instructions

## 🔹 Run the Program

```bash
python src/main.py
```

Or specify a custom network file:

```bash
python src/main.py --input network1.xlsx --method darcy
```

### Available Head-Loss Methods

- `--method hazen`
- `--method darcy`

---

## 🔹 Output

After convergence, the program:

- Prints final balanced pipe flows
- Saves results to `/results/`

Example Output:

```
Converged after 6 iterations
R² = 0.999
RMSE = 0.0008
MBE = 0.0000
```

---

# 🔄 Algorithm Workflow

1. Read Excel input  
2. Verify continuity at nodes  
3. Assume initial flow distribution  
4. Compute head losses  
5. Apply loop correction ΔQ  
6. Update flows  
7. Check convergence tolerance  
8. Repeat until ∑hf ≈ 0  

---

# 📐 Governing Equations

### Continuity

∑ Q_in = ∑ Q_out

### Energy Conservation

∑ h_f = 0

### Supported Head-Loss Models

- Hazen–Williams  
- Darcy–Weisbach  

---

# 📈 Future Improvements

- Graphical User Interface (GUI)
- Reservoir and pump modeling
- Real-time simulation
- Web-based visualization dashboard
- Integration with larger hydraulic solvers

---

# 🎓 Academic Contribution

This tool is suitable for:

- Civil Engineering students
- Hydraulic design engineers
- Research applications
- Teaching computational hydraulics

It bridges traditional hydraulic theory with modern scientific programming.

<!-- ---

# 📚 Citation

```
Iyeke, S.D., & Moses, G.O. (2025).
Computer Programming Approach to Hydraulic Flow Balancing
in Looped Networks Using Hardy Cross Method.
Journal of Energy Technology and Environment, 6(3), 30–38.
```

--- -->

# 👨‍💻 Author

Developed as part of a published hydraulic modeling research project integrating civil engineering theory with computational implementation.

