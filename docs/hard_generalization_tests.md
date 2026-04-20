# Hard Generalization Tests for MAPPO

## Current Weaknesses Identified
- **Spatial scaling**: 16x16 grid → 50% success (vs 100% at 8x8)
- **Evader-seeking rate drops**: 45% on large grids (vs 77% baseline)
- **Takes 10x more steps**: 113 steps vs 12 steps on larger grids

---

## Recommended Hard Tests (Ranked by Expected Difficulty)

### 🔴 **EXTREME DIFFICULTY**

#### 1. **Asymmetric Grid (Rectangular)**
```python
x_size=16, y_size=8, max_cycles=300
```
**Why hard**: Trained on square grids. Breaks spatial reasoning assumptions.
**Prediction**: <40% success rate

#### 2. **Very Large Grid (32x32)**
```python
x_size=32, y_size=32, max_cycles=1000
```
**Why hard**: 16x larger area, observation window still 7x7
**Prediction**: <20% success rate

#### 3. **Many Evaders + Moving (3+ evaders, unfrozen)**
```python
n_pursuers=2, n_evaders=3, freeze_evaders=False, max_cycles=200
```
**Why hard**: Multiple moving targets, coordination required
**Prediction**: <30% success rate

#### 4. **Obstacle-Rich Environment (custom map)**
```python
# Would require modifying pursuit_v4 to add obstacles
# Dense wall patterns that require path planning
```
**Why hard**: Trained on relatively open spaces
**Prediction**: <35% success rate

#### 5. **Long Horizon + Sparse Reward**
```python
x_size=20, y_size=20, max_cycles=2000, n_catch=2
```
**Why hard**: Credit assignment over very long episodes
**Prediction**: <25% success rate

---

### 🟡 **MODERATE-HIGH DIFFICULTY**

#### 6. **Surround Capture Mode**
```python
n_pursuers=4, n_evaders=1, surround=True, n_catch=4, freeze_evaders=True
```
**Why hard**: Different capture mechanics (surround vs overlap)
**Prediction**: 40-60% success rate

#### 7. **Asymmetric Team (4 pursuers, 1 needs to catch)**
```python
n_pursuers=4, n_evaders=1, n_catch=1, freeze_evaders=True, x_size=12, y_size=12
```
**Why hard**: Overcrowding, collision avoidance critical
**Prediction**: 50-70% success rate

#### 8. **Mixed Scenario (some moving, some frozen)**
```python
n_evaders=2, freeze_evaders=False  # But modify to freeze 1 of 2
```
**Why hard**: Heterogeneous target behavior
**Prediction**: 60-75% success rate

#### 9. **Partial Observability Increased**
```python
# Modify observation radius from 7x7 to 5x5
```
**Why hard**: Less information for coordination
**Prediction**: 70-80% success rate

#### 10. **Higher n_catch threshold**
```python
n_pursuers=2, n_catch=3  # Impossible without help from evaders?
# Or: n_pursuers=3, n_catch=3, but trained on n_catch=2
```
**Why hard**: Different capture dynamics
**Prediction**: Variable (0-100% depending on config)

---

### 🟢 **MODERATE DIFFICULTY (Still Interesting)**

#### 11. **Slightly Larger Grid (12x12)**
```python
x_size=12, y_size=12, max_cycles=250
```
**Why hard**: Intermediate scaling test
**Prediction**: 70-85% success rate

#### 12. **Many Pursuers, Frozen Evaders (5+ pursuers)**
```python
n_pursuers=5, n_evaders=1, n_catch=2, freeze_evaders=True
```
**Why hard**: Overcrowding without collision modeling
**Prediction**: 75-90% success rate

#### 13. **Shared Reward Mode**
```python
shared_reward=True
```
**Why hard**: Different reward structure (trained on individual rewards)
**Prediction**: 80-95% success rate

---

## Implementation Priority

### **Immediate Tests (Easiest to Implement)**
1. ✅ 12x12 grid (small code change)
2. ✅ 32x32 grid (small code change)  
3. ✅ Asymmetric grid 16x8 (small code change)
4. ✅ Surround mode (parameter change)
5. ✅ Many pursuers (5-6) (parameter change)
6. ✅ 3 moving evaders (parameter change)

### **Scripts to Create**
```bash
eval_mappo_grid_12x12.py          # Moderate scaling
eval_mappo_grid_32x32.py          # Extreme scaling
eval_mappo_asymmetric_16x8.py     # Asymmetric grid
eval_mappo_surround_mode.py       # Different capture mechanics
eval_mappo_overcrowding_5p.py     # Many pursuers
eval_mappo_many_moving_evaders.py # 3+ moving evaders
```

---

## Expected Results Summary

| Scenario | Predicted Success | Rationale |
|----------|-------------------|-----------|
| **32x32 grid** | 10-20% | Spatial scale breaks down |
| **16x8 asymmetric** | 30-40% | Square assumption violated |
| **3 moving evaders** | 25-35% | Multiple targets + movement |
| **12x12 grid** | 70-85% | Moderate scaling |
| **Surround mode** | 40-60% | Different game rules |
| **5 pursuers** | 75-90% | Overcrowding issues |

---

## Additional Research Questions

1. **Does observation window size matter?**
   - Test with modified observation radius (5x5 vs 7x7)

2. **Does training grid size affect generalization slope?**
   - Train on 10x10, test on 8x8, 12x12, 16x16, 20x20
   - Measure generalization curve

3. **Can curriculum learning help?**
   - Progressive training: 8x8 → 10x10 → 12x12
   - Test if it improves 16x16 performance

4. **Is the centralized critic helping or hurting?**
   - MAPPO uses global state for value function
   - Does this create overfitting to grid size?

5. **What about transfer across different mechanics?**
   - Train with surround=True, test with surround=False
   - Train with shared_reward=True, test with shared_reward=False
