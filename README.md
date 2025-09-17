## The IEEE 30-bus Test Case
It’s a **benchmark power system** model, originally published by the IEEE (Institute of Electrical and Electronics Engineers), that represents a *small but realistic transmission grid*. It is a simple approximation of a section of the American power grid as it was in 1961. Researchers and engineers use it to:
- Test **power flow algorithms**.
- Validate and compare **ACOPF/DCOPF solvers**.
- Benchmark **optimization, machine learning, or control methods**.

### Its Components
This benchmark system comprises:
- **30 buses (nodes)**: These are connection points — either loads, generators, or junctions.
- **41 transmission lines/branches**: They connect buses, each with reactance, resistance, and line limits.
- **6 generators**: Located at specific buses (e.g., bus 1 is the slack/reference bus). They supply real and reactive power within limits.
- **Loads**: Many buses represent demand, specified in MW (real power) and Mvar (reactive power).
- **Shunts**: Some buses include shunt elements (like capacitors) to support voltage.

So it can be pictured as a **graph** with 30 nodes and 41 edges, but where each node has electrical variables (voltage, angle, load, generation).

### Simulating the IEEE 30-bus Test Case on MATLAB
This benchmark system can be simulated on MATLAB using MATPOWER, an open-source MATLAB/Octave package for **power system simulation and optimization**. Think of it as the **framework** (like PyTorch or TensorFlow) that loads the dataset and runs baseline solvers.

It’s widely used by researchers and utilities for:
- Solving **power flow (PF)** problems.
- Running **optimal power flow (OPF)** (both AC and DC).
- Benchmarking new algorithms.

Think of it as a “toolbox” where you load a test case (like IEEE 30-bus) and run solvers. Using MATPOWER, you can run PF, OPF (AC and DC) to solve the power flow equations


### **The Dataset**
Found inside the [matpower_format](matpower_format/case30) folder

#### **`Train` set**
* `X`: Contains the inputs to our models. Has shape `(num_samples, n_buses, num_features)` $= (600, 30, 4)$. The power system is based on the IEEE case30 format with 30 buses and 6 generators in the grid, more information on this grid can be found at [IEEEcase30 format info](https://matpower.org/docs/ref/matpower5.0/case_ieee30.html). The `X` in the validation (`val`) and test (`test`) sets' also have the same features but the number of samples differs. The 4 features are:
    - `Pd`: Real power demand
    - `Qd`: Reactive power demand
    - `mag_Sd`: Magnitude of apparent power demand
    - `ang_Sd`: Phase angle of apparent power demand

 $$S_d = P_d + jQ_d$$

* `A`: Contains the sparse adjacency matrices for each sample. It has shape `(num_samples, axes, num_nonzero)` $= (600, 2, 82)$ indicating that the each adjacency has 82 nonzero elements. `dim` is always 2 because a matrix has two axes i.e., rows and columns. For `A_val` and `A_test`, the only difference is the number of samples.

* `Y`: Contains the targets/outputs from our models. In this case, the targets are the voltage magnitude (`Vm`) and voltage phase angle (`Va`) from the AC optimal power flow results. This array has shape `(num_samples, num_buses, num_targets)` $= (600, 30, 2)$. The 2 targets are `Vm` and `Va`.

* `B`: Contains additional information for each grid sample. Each sample's information is stored in a dictionary with these keys:
    - `gencost`: Data related to cost curves of generators. (6,7)
    - `gen_indx`: Indices of active generators in the system mapped to the bus numbers where they are connected. (6,)
    - `Pg`: Real power generation output for each generator from the ACOPF solution. (6,)
    - `Qg`: Reactive power generation output for each generator from the ACOPF solution. (6,)
    - `MAX_Pg`: Maximum real power generation limit for each generator. (6,)
    - `MAX_Qg`: Maximum reactive power generation limit for each generator. (6,)
    - `MIN_Pg`: Minimum real power generation limit for each generator. (6,)
    - `MIN_Qg`: Minimum reactive power generation limit for each generator. (6,)
    - `f`: **Objective function** value from the ACOPF solution. (1,1)
    - `branch`: Subset of the branch data. The selected columns `([0,1,5,11,12])` are the most relevant parameters e.g., "from" and "to" bus numbers, etc. (41, 5)
    - `V-dcopf`: Voltage magnitude and angle at each bus from the DC optimal power flow (DCOPF) solution. (30, 2)
    - `S-dcopf`: The complex power generation (real and reactive) for each generator from the DCOPF solution. (6, 2)
    - `f-dcopf`: Holds the objective function value from the DCOPF solution. (1, 1)
    - `Ybus`: Bus admittance matrix. (30, 30)
    - `Yt`: Component of the Ybus matrix related ot the **"to" buses** of the branches. (41, 30)
    - `Yf`:  Component of the Ybus matrix related ot the **"from" buses** of the branches. (41, 30)

 **NOTE**
 
 It might seem counterintuitive that `Yt` and `Yf` with a shape of (41, 30) are components of `Ybus` with a shape of (30, 30), but this is due to how these matrices are constructed in power system analysis.

 The elements of `Ybus` represent the admittances between buses and the shunt admittances at each bus. `Yf` and `Yt` are not direct sub-matrices of `Ybus`. Instead, they are matrices that represent the contribution of each branch to the injected currents at the "from" and "to" buses, respectively, when multiplied by the bus voltage vector. 

 * `Yf` (shape 41x30): Each row corresponds to a branch (41 branches). The columns correspond to the buses (30 buses). When you multiply `Yf` by the bus voltage vector (30x1), the resulting vector (41x1) gives you the currents flowing from the "from" bus into each branch.
 * `Yt` (shape 41x30): Similar to `Yf`, each row corresponds to a branch, and the columns correspond to the buses. When you multiply `Yt` by the bus voltage vector (30x1), the resulting vector (41x1) gives you the currents flowing from each branch into the "to" bus.

 The `Ybus` matrix is actually constructed using `Yf`, `Yt`, and the branch admittance matrix (`Ybr`), which represents the series admittance of each branch.

 For `gencost` the format is the same as that used in [matpower case format](https://matpower.org/docs/ref/matpower5.0/caseformat.html)