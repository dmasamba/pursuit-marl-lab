# MAPPO Generalization Analysis Summary

## 📊 Current Performance Results

### Baseline Training Environment
- **Grid**: 8x8 (64 cells)
- **Agents**: 2 pursuers, 1 evader (frozen)
- **Success**: 100% ✅
- **Steps**: 11.6 avg
- **Invalid actions**: 2.4%

---

## 🔍 Generalization Test Results

| Test | Success Rate | Δ from Baseline | Steps | Key Finding |
|------|--------------|-----------------|-------|-------------|
| **Baseline (8x8, 2P, 1E, frozen)** | 100% | — | 11.6 | Perfect control |
| **3 Pursuers (all must catch)** | 93% | ↓7% | 15.0 | Coordination challenge |
| **2 Evaders** | 98% | ↓2% | 21.1 | Handles multi-target well |
| **Moving Evaders** | 100% | ±0% | 12.6 | **Surprising: no degradation!** |
| **16x16 Grid (4x area)** | **50%** | ↓**50%** | 112.8 | **Major failure mode** |

---

## 🎯 Critical Insights

### 1. **Spatial Scaling = Weak Point** 🔴
- 16x16 grid (4x larger) → 50% success rate
- 10x more steps required
- Evader-seeking drops from 77% to 45%
- **Root cause**: Fixed 7x7 observation window doesn't scale

### 2. **Movement Invariance** 🟢
- Moving evaders: 100% success (!)
- Trained on frozen, works on moving
- **Why**: Local observation captures relative motion well

### 3. **Team Scaling is OK** 🟡
- 3 pursuers: 93% success
- Modest degradation with team size
- Coordination still functional

### 4. **Multi-Target Handling** 🟢
- 2 evaders: 98% success
- Slightly longer episodes (21 steps vs 12)
- Policy generalizes to multiple targets

---

## 🚀 Recommended Hard Tests

### **Top 5 Expected Failure Modes:**

#### 1️⃣ **32x32 Grid** (Extreme Scaling)
```bash
python eval_mappo_grid_32x32.py
```
- **Prediction**: 10-20% success
- **Why**: 16x larger area, observation coverage insufficient
- **Script**: ✅ Created

#### 2️⃣ **Asymmetric 16x8 Grid** (Shape Change)
```bash
python eval_mappo_asymmetric_16x8.py
```
- **Prediction**: 30-40% success
- **Why**: Trained on square, breaks spatial assumptions
- **Script**: ✅ Created

#### 3️⃣ **3 Moving Evaders** (Multi-Target + Movement)
```bash
python eval_mappo_many_moving_evaders.py
```
- **Prediction**: 25-35% success
- **Why**: Coordination + tracking multiple moving targets
- **Script**: ⏳ To be created

#### 4️⃣ **Surround Capture Mode** (Different Mechanics)
```bash
python eval_mappo_surround_mode.py
```
- **Prediction**: 40-60% success
- **Why**: Different capture rules (surround vs overlap)
- **Script**: ⏳ To be created

#### 5️⃣ **12x12 Grid** (Moderate Scaling Baseline)
```bash
python eval_mappo_grid_12x12.py
```
- **Prediction**: 70-85% success
- **Why**: Intermediate test between 8x8 and 16x16
- **Script**: ⏳ To be created

---

## 📈 Generalization Curve Hypothesis

```
Success Rate
100% |●────────────╲
     |              ╲
 75% |               ●
     |                ╲
 50% |                 ●
     |                  ╲
 25% |                   ●
     |                    ╲
  0% |                     ●
     +─────────────────────────
       8x8  10x10  12x12  16x16  32x32
              Grid Size
```

**Hypothesis**: Exponential degradation with spatial scale
- 8x8: 100%
- 12x12: ~75% (predicted)
- 16x16: 50% (confirmed)
- 32x32: <20% (predicted)

---

## 🧠 Why MAPPO Fails at Large Scales

### Architectural Issues:
1. **Fixed receptive field** (7x7 obs window)
   - Covers 6% of 8x8 grid ✅
   - Covers 1.5% of 16x16 grid ❌
   - Covers 0.4% of 32x32 grid ❌❌

2. **Centralized critic overfitting**
   - Value network sees (N, H, W, C) global state
   - Likely memorized 8x8 spatial patterns
   - Doesn't generalize to new dimensions

3. **No explicit spatial reasoning**
   - CNN learns position-dependent features
   - No coordinate encoding or attention
   - Navigation degrades without full visibility

### Behavioral Breakdown:
- **Evader-seeking drops**: 77% → 45% on 16x16
- **Random wandering**: High steps (112 vs 12)
- **Invalid actions stable**: ~3% (collision avoidance works)

---

## 💡 Potential Solutions

### Short-term (Easy):
1. ✅ Increase max_cycles for larger grids
2. ✅ Test intermediate sizes (10x10, 12x12)
3. ⏳ Train on mixed-size grids (curriculum)

### Medium-term (Research):
1. **Attention mechanisms** for variable-size grids
2. **Coordinate encoding** (add x,y position to observations)
3. **Graph neural networks** (agent-evader relationships)
4. **Hierarchical policies** (high-level navigation + low-level control)

### Long-term (Ambitious):
1. **Foundation models** pre-trained on diverse grids
2. **Meta-learning** for fast adaptation
3. **World models** for planning at scale

---

## 📝 Next Steps

### Priority 1: Complete Hard Test Suite
- [ ] Run 32x32 grid test
- [ ] Run asymmetric 16x8 test
- [ ] Create and run 12x12 test
- [ ] Create and run 3 moving evaders test
- [ ] Create and run surround mode test

### Priority 2: Analysis
- [ ] Plot generalization curve (success vs grid size)
- [ ] Analyze evader-seeking rate degradation
- [ ] Visualize failure modes (trajectories)
- [ ] Compare to IPPO/Shared baselines

### Priority 3: Improvements
- [ ] Train on curriculum (8x8 → 10x10 → 12x12)
- [ ] Add coordinate encoding
- [ ] Test with attention mechanisms
- [ ] Ablation: fixed vs adaptive observation radius

---

## 🎓 Research Implications

**Key Finding**: MAPPO's generalization is primarily limited by **spatial scale**, not:
- ❌ Team size changes
- ❌ Number of targets
- ❌ Target movement

**Hypothesis**: Local policies + centralized critics are robust to **game-theoretic changes** but fragile to **environmental scale changes**.

**Why this matters**: 
- Multi-agent RL often evaluated on fixed environments
- Spatial generalization is crucial for real-world deployment
- Current architectures may need fundamental redesign for scale

---

## 📚 Files Created

1. ✅ `hard_generalization_tests.md` - This analysis document
2. ✅ `eval_mappo_grid_32x32.py` - Extreme scaling test (32x32)
3. ✅ `eval_mappo_asymmetric_16x8.py` - Asymmetric grid test
4. 📁 `artifacts/` - Organized experiment outputs

**All scripts include**:
- Comprehensive metrics tracking
- Timestamped output directories
- CSV + JSON results
- Consistent evaluation protocol
