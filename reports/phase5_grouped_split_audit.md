# Phase 5 Deliverable: Drive-Level Split Leakage Audit and Grouped Partitioning

**Date:** October 2026  
**Auditor:** Senior Research Engineer & Inertial Navigation Specialist  
**Deliverable Document:** [`reports/phase5_grouped_split_audit.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase5_grouped_split_audit.md)  
**Associated Artifacts:**
- Grouped Split Manifest: [`data/splits/drive_split_manifest_grouped.json`](file:///c:/Users/shreya%20parmar/Downloads/research/data/splits/drive_split_manifest_grouped.json)
- Grouped Tensors: `data/processed/*_windows_grouped.npz`
- Grouped Normalizer: `data/processed/sensor_normalizer_grouped.json`

---

## 1. Executive Summary & Forensic Discovery

The previous experimental configuration partitioned Driver A's recordings by assigning:
- **Validation:** `S-S3a.csv`
- **Testing:** `S-S3b.csv` and `S-S3c.csv`

Forensic analysis of geodetic coordinates, timestamps, and route continuity reveals that **`S3a`, `S3b`, and `S3c` are contiguous fragments of the exact same 2-hour out-and-back journey on Wednesday, September 4, 2019**:
- **`S3a` (Outbound leg):** Coventry $\to$ Rugby ($19:21:51 - 20:02:53$). Ends at $(52.37075^\circ, -1.25435^\circ)$.
- **`S3b` (Mid-point loop):** Starts at $(52.37046^\circ, -1.25443^\circ)$ **just 10 seconds later** ($20:03:03 - 20:14:29$) at the exact same location ($\Delta p < 20\text{ m}$).
- **`S3c` (Return leg):** Starts at $(52.37941^\circ, -1.26178^\circ)$ **just 4 seconds later** ($20:14:28 - 21:16:26$) and returns directly to the starting point of `S3a` in Coventry ($52.40568^\circ, -1.50011^\circ$).

Splitting `S3a` into validation while evaluating `S3b` and `S3c` in testing constituted **severe temporal, spatial, and driver route leakage**. The validation and test sets were operating on the exact same roads, in the exact same vehicle, with the exact same driver, separated by mere seconds.

---

## 2. Complete Forensic Timeline of All IO-VNBD Drives

Analyzing the full dataset reveals exactly **three distinct independent driving sessions/trips** recorded on different calendar days:

| Session ID | Recording Files Included | Calendar Date | Driving Time Range (BST) | Duration (min) | Geographic Route Description | Independence Status |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Session 1** | `S-S3a`, `S-S3b`, `S-S3c` | **Sept 4, 2019** | $19:21 - 21:16$ | 114.6 min | Coventry to Rugby out-and-back round trip | **Single Connected Journey** |
| **Session 2** | `S-S4` | **Sept 6, 2019** | $18:16 - 20:59$ | 157.7 min | Coventry urban & suburban ring road network | **Completely Independent Day** |
| **Session 3** | `S-S1`, `S-S2` | **Sept 8, 2019** | $10:07 - 14:40$ | 242.7 min | Coventry to Nuneaton morning & afternoon legs | **Single Connected Sunday Journey** |

---

## 3. Scientifically Defensible Grouped Partitioning

To guarantee zero leakage, every fragment of an underlying journey must be held together within the same partition:

### Grouped Partition Assignment:
- **TRAINING PARTITION (Session 3):** `S-S1.csv` and `S-S2.csv`  
  - Total Samples: 145,533 samples ($\sim 4.04\text{ hours}$).  
  - Sliding Windows: **14,525 windows** ($W=100$, stride=10).  
  - Covers suburban arterial roads and multi-lane expressways.
- **VALIDATION PARTITION (Session 1):** `S-S3a.csv`, `S-S3b.csv`, `S-S3c.csv`  
  - Total Samples: 68,495 samples ($\sim 1.90\text{ hours}$).  
  - Sliding Windows: **6,812 windows**.  
  - Used strictly for model selection, learning-rate annealing, and early stopping.
- **TEST PARTITION (Session 2):** `S-S4.csv`  
  - Total Samples: 91,451 samples ($\sim 2.54\text{ hours}$).  
  - Sliding Windows: **9,117 windows**.  
  - **Completely independent holdout journey** recorded on September 6, 2019. Held strictly untouched until final evaluation.

### Normalization Isolation Guarantee:
The feature normalizer (`sensor_normalizer_grouped.json`) is fitted **strictly on the 145,533 samples of the Training Partition (Session 3)**. Validation and test partitions are transformed using frozen parameters.

---

## 4. Evaluation of Fixed Holdout vs. Grouped Cross-Validation

Given that the dataset contains exactly 3 multi-hour driving sessions, there is a clear trade-off:
1. **Primary Benchmark (Fixed Session Holdout):**
   - Train on Session 3 (4.04h), Validate on Session 1 (1.90h), Test on Session 2 (2.54h).
   - *Advantage:* Provides an untouched, frozen test set (`S4`) that mirrors real-world deployment where a model is evaluated on a completely unseen day and route.
2. **3-Fold Grouped Cross-Validation Across Sessions:**
   - Fold 1: Train on Sessions 2 & 3, Validate on Session 1.
   - Fold 2: Train on Sessions 1 & 3, Validate on Session 2.
   - Fold 3: Train on Sessions 1 & 2, Validate on Session 3.
   - *Advantage:* Eliminates variance from any single route's idiosyncrasies.
   - *Decision:* We freeze Option 1 (Fixed Session Holdout with `S4` as Test) as our primary benchmark, with 3-fold grouped CV documented as an optional verification protocol.

*Phase 5 Split Leakage Audit complete. Proceeding to Phase 6 (GPS-Outage Estimator & Causality).*
