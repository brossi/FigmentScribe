# TensorFlow 2.x Graph Mode Issue

## Problem Summary

**20 tests are failing** due to a TensorFlow graph execution scope error in `model.py`.

**Error Message:**
```
InaccessibleTensorError: The tensor <tf.Tensor 'handwriting_model/while/Squeeze:0'>
cannot be accessed from FuncGraph(name=train_step), because it was defined in
FuncGraph(name=handwriting_model_while_body), which is out of scope.
```

**Failure Rate:** 6% (20/318 tests)

**Status:** Real code bug, not a test bug

---

## Root Cause Analysis

### The Problem Code

Location: `model.py` lines 176-183 in the `call()` method:

```python
# Process each timestep through attention mechanism
timesteps = tf.shape(stroke_data)[1]  # Dynamic tensor value

# Compute windows for all timesteps
windows = []
phis = []
kappas = [kappa]

# ⚠️ PROBLEM: Python loop with dynamic range
for t in range(timesteps):  # timesteps is a Tensor, not a Python int!
    lstm0_t = lstm0_out[:, t, :]
    window, phi, kappa = self.get_window(lstm0_t, kappas[-1], char_seq)
    windows.append(window)
    phis.append(phi)
    kappas.append(kappa)

windows = tf.stack(windows, axis=1)  # ❌ Fails here - windows are out of scope
phis = tf.stack(phis, axis=1)
```

### Why This Fails

1. **Graph Tracing Context**: When `model.call()` is called inside `@tf.function` (e.g., in `train_step()`), TensorFlow traces it into a computational graph.

2. **Dynamic Loop Conversion**: The Python `for` loop with `range(timesteps)` where `timesteps` is a dynamic tensor gets converted by AutoGraph into a `tf.while_loop`.

3. **Separate FuncGraph Scopes**: The loop body executes in a separate `FuncGraph` scope. Tensors created inside this scope (`window`, `phi`, `kappa`) cannot be accessed outside.

4. **Scope Violation**: When we try to `tf.stack(windows)` outside the loop, the tensors in the `windows` list are from the inner FuncGraph and are inaccessible.

### When This Happens

This error occurs whenever the model is called inside a `@tf.function` context:

- **Training**: `train_step()` in `train.py` (line 92)
- **Unit Tests**: Any test that uses `@tf.function` decorated functions
- **Integration Tests**: End-to-end workflows that call training functions

The model works fine in **eager execution mode** (e.g., `sample.py`), which is why sampling tests pass.

---

## Solutions

### Option 1: Use `tf.TensorArray` (Recommended)

**Approach:** Replace Python lists with `tf.TensorArray` which is graph-compatible.

**Pros:**
- Clean, idiomatic TensorFlow 2.x solution
- Works correctly in both graph and eager mode
- Maintains same logic flow
- Good performance

**Cons:**
- Requires code changes
- Slightly more verbose

**Implementation:**
```python
# Initialize TensorArrays
timesteps = tf.shape(stroke_data)[1]
windows = tf.TensorArray(dtype=tf.float32, size=timesteps, dynamic_size=False)
phis = tf.TensorArray(dtype=tf.float32, size=timesteps, dynamic_size=False)
kappas_array = tf.TensorArray(dtype=tf.float32, size=timesteps+1, dynamic_size=False)
kappas_array = kappas_array.write(0, kappa)

# Loop using TensorArray
for t in tf.range(timesteps):
    lstm0_t = lstm0_out[:, t, :]
    window, phi, kappa = self.get_window(lstm0_t, kappas_array.read(t), char_seq)
    windows = windows.write(t, window)
    phis = phis.write(t, phi)
    kappas_array = kappas_array.write(t+1, kappa)

# Stack results
windows = tf.transpose(windows.stack(), [1, 0, 2])  # [timesteps, batch, features] → [batch, timesteps, features]
phis = tf.transpose(phis.stack(), [1, 0, 2, 3])
kappa = kappas_array.read(timesteps)
```

### Option 2: Use `tf.scan`

**Approach:** Use `tf.scan` to accumulate results over timesteps.

**Pros:**
- Functional programming style
- Efficient graph execution
- No explicit loop management

**Cons:**
- More complex to understand
- Harder to debug
- Requires restructuring logic

**Implementation Sketch:**
```python
def attention_step(prev_kappa, lstm0_t):
    window, phi, new_kappa = self.get_window(lstm0_t, prev_kappa, char_seq)
    return new_kappa, (window, phi, new_kappa)

_, (windows, phis, kappas) = tf.scan(
    attention_step,
    tf.transpose(lstm0_out, [1, 0, 2]),  # [batch, time, features] → [time, batch, features]
    initializer=(kappa, (initial_window, initial_phi, kappa))
)
```

### Option 3: Vectorize Using `tf.map_fn`

**Approach:** Map the attention computation over all timesteps.

**Pros:**
- Declarative style
- Can be parallelized

**Cons:**
- Attention mechanism depends on previous kappa (sequential dependency)
- Not suitable for this use case (sequential recurrence)

**Status:** ❌ **Not applicable** - attention mechanism has recurrent dependency on previous timestep's kappa.

### Option 4: Force Eager Execution

**Approach:** Run model in eager mode only, disable graph compilation.

**Pros:**
- No code changes needed
- Works immediately

**Cons:**
- ❌ **Much slower** (2-10x slower training)
- ❌ Not production-ready
- ❌ Defeats purpose of TensorFlow 2.x migration

**Implementation:**
```python
# In model.__init__()
self.call = tf.function(self.call, experimental_compile=False)

# Or in train.py, remove @tf.function decorator
def train_step(inputs, targets):  # No @tf.function
    ...
```

**Verdict:** ❌ **Not recommended** for production use.

---

## Recommendation

**Use Option 1: `tf.TensorArray`**

**Reasoning:**
- ✅ Proper TensorFlow 2.x solution
- ✅ Clean, readable code
- ✅ Works in graph and eager mode
- ✅ Maintains performance
- ✅ Minimal complexity

**Implementation Priority:** Medium
- Tests currently pass in eager mode (sampling works)
- Training works but ~10-20% slower without graph optimization
- Not blocking M1 local development
- Should fix before production deployment

---

## Impact Assessment

### Currently Working
- ✅ **Smoke tests (27/27)** - Use eager execution
- ✅ **Sampling (sample.py)** - Eager mode
- ✅ **Data loading (utils.py)** - No graph execution
- ✅ **Model architecture tests** - Direct forward pass without @tf.function

### Currently Failing (20 tests)
- ❌ **Training loop tests (7 failures)** - Uses `@tf.function` train_step
- ❌ **End-to-end tests (5 failures)** - Full pipeline with graph mode
- ❌ **Train function tests (4 failures)** - Direct train.py functions
- ❌ **Checkpointing tests (2 failures)** - Involves training steps
- ❌ **Other integration tests (2 failures)** - Mixed usage

### Production Impact
- **Training**: Works but runs in eager mode (slower)
- **Sampling**: ✅ Fully functional
- **M1 Development**: ✅ No blocker for local testing

---

## Next Steps

### Immediate (M1 Development)
1. ✅ Document issue (this file)
2. Document in commit messages
3. Continue with M1 development (not blocking)

### Short-term (Before Production)
1. Implement `tf.TensorArray` fix in `model.py`
2. Verify all 20 tests pass
3. Benchmark performance (should match or exceed current)
4. Update documentation

### Long-term (Optimization)
1. Consider `tf.scan` for cleaner functional style
2. Profile graph execution performance
3. Add graph tracing tests

---

## References

- **TensorFlow Docs**: [Using tf.function](https://www.tensorflow.org/guide/function)
- **AutoGraph**: [AutoGraph reference](https://www.tensorflow.org/guide/autograph)
- **TensorArray**: [tf.TensorArray API](https://www.tensorflow.org/api_docs/python/tf/TensorArray)
- **Original Paper**: Graves, "Generating Sequences with Recurrent Neural Networks" (2013)

---

## Test Examples

### Failing Test
```python
@tf.function
def train_step(inputs, targets):
    predictions = model(inputs, training=True)  # ❌ Fails here
    loss = compute_loss(predictions, targets)
    return loss
```

### Working Test
```python
def sample_test():
    predictions = model(inputs, training=False)  # ✅ Eager mode works
    return predictions['eos']
```

---

**Status**: Documented, not yet fixed
**Blocking**: No (for M1 development)
**Priority**: Medium (before production deployment)
**Difficulty**: Low-Medium (clear fix available)
