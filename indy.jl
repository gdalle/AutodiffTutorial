### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 028bee0c-00d6-11ef-2159-99ffeebdeba2
begin
	using CairoMakie
	import ForwardDiff
	using LinearAlgebra
	using PlutoTeachingTools
	using PlutoUI
	using SparseArrays
	import Symbolics
	import Zygote
end

# ╔═╡ cbf0a7eb-0750-4537-b925-efd743360979
TableOfContents(depth=2)

# ╔═╡ 2fe59b4a-3bd4-4466-83ff-aa97a887b1b3
md"""
# A primer on algorithmic differentiation
"""

# ╔═╡ fe638244-3f35-4de4-a52e-aa6ef8b662ef
TwoColumn(md"**Guillaume Dalle**", md"INDY Lab, 2024.04.27")

# ╔═╡ d98a6acc-8cb9-4ec3-beb2-c12655f0bd72
md"""
!!! tip "Link to the notebook"
	<https://gdalle.github.io/AutodiffTutorial/>
"""

# ╔═╡ 824423d7-21b0-4b48-9200-22d213207fe0
ChooseDisplayMode()

# ╔═╡ 603d7e48-3def-44f8-882e-74e3f22293c0
md"""
# Differentiable programming
"""

# ╔═╡ fdb8ed44-74dd-4cdf-8d3d-c4d8bf467fc4
md"""
## In theory
"""

# ╔═╡ 317749dd-fd49-488b-ad0d-9d30b33d9891
md"""
### Definition

_**Differentiable programming** is a programming paradigm in which complex computer programs (including those with control flows and data structures) can be **differentiated end-to-end automatically**, enabling gradient-based optimization of parameters in the program.
In differentiable programming, a program is also defined as the composition of elementary operations, forming a **computation graph**._

Blondel and Roulet (2024)
"""

# ╔═╡ 7b023270-9b2d-4d69-93dc-921e37fcf423
md"""
### Computation graphs
"""

# ╔═╡ ff41d45f-12d8-41de-bf19-e290a0590098
LocalResource("images/dag.png")  # source: "The Elements of Differentiable Programming"

# ╔═╡ b7227f5c-bf16-460d-b2f2-6938a2ca8261
md"""
## In practice
"""

# ╔═╡ 7b64bd44-f261-49b6-8520-ec09135305d3
md"""
### Applications
"""

# ╔═╡ 52de1ea5-e8f0-4efb-a752-fdddf29ff462
md"""
- Deep learning
- Probabilistic programming
- Scientific computing
"""

# ╔═╡ 5dda51ca-cf43-4ff5-931c-258ed1268a4e
md"""
### Python implementations
"""

# ╔═╡ 61d3efb7-39e8-4c4b-ad8b-fd3920e8457c
md"""
Choose AD first, write code second.

- [JAX](https://github.com/google/jax)
- [PyTorch](https://github.com/pytorch/pytorch)
- [TensorFlow](https://github.com/tensorflow)
"""

# ╔═╡ 5198f895-9bf4-4f5c-ad56-109f856f4f06
md"""
### Julia implementations
"""

# ╔═╡ 07cb8a3f-2627-47ad-bccc-abf06b994787
md"""
Write code first, choose AD second (and maybe adapt code).

Many different options, see [juliadiff.org](https://juliadiff.org/) for a list

Unification and comparison attempt: my project [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl)
"""

# ╔═╡ ebf1147e-4469-4b22-99d2-003c3d597890
md"""
## References
"""

# ╔═╡ a2030c95-7ed4-4690-9a22-3935f7f7ab47
md"""
> [The Elements of Differentiable Programming](https://arxiv.org/abs/2403.14606), Blondel and Roulet (2024)
>
> [Automatic Differentiation in Machine Learning: A Survey](http://jmlr.org/papers/v18/17-468.html), Baydin et al. (2018)
>
> [Evaluating derivatives: principles and techniques of algorithmic differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717761), Griewank and Walther (2008)
"""

# ╔═╡ 1298e858-8586-4589-b853-c7165a5326ed
md"""
# The 4 horsemen of differentiation
"""

# ╔═╡ 2d48dff9-cf14-4ede-9340-8c2d813ef8b8
md"""
Here is a simple test function. What does it compute?
"""

# ╔═╡ 086435e8-2d8e-49a6-a2ec-505076f0d35b
x0 = 2.0

# ╔═╡ 129bffac-2676-453e-a8bb-404649005625
sqrt(x0)

# ╔═╡ bc8eda8e-3d2d-4939-9e3c-e0cb122fa7e8
md"""
## Horseman I: Manual
"""

# ╔═╡ a3c02f24-2eb6-4895-bfd5-e95a26ac44fe
md"""
!!! info "Principle"
	Write down derivatives by hand

!!! danger "Downside"
	Labor-intensive, not adaptive

- [Leibniz rule](https://en.wikipedia.org/wiki/General_Leibniz_rule)
- [Faà di Bruno's formula](https://en.wikipedia.org/wiki/Fa%C3%A0_di_Bruno%27s_formula)
"""

# ╔═╡ 4efaf69a-ecb3-47ad-9d00-60cd6e49d1b4
md"""
## Horseman II: Symbolic
"""

# ╔═╡ 8b4aa562-9443-40ee-8d52-c85d5cb1ef81
md"""
!!! info "Principle"
	Ask Wolfram Alpha (or an open source alternative) for the exact formula

!!! danger "Downside"
	Expression swell as the computation graph is flattened
"""

# ╔═╡ cc73da3a-27ce-4963-9e28-775b34d70c29
md"""
 $L =$ $(@bind L_slider PlutoUI.Slider(0:7; show_value=true))
"""

# ╔═╡ f5bdc5ca-a291-4244-91e1-80f6a591e5af
md"""
## Horseman III: Numerical
"""

# ╔═╡ b4f82186-70ae-4f4f-9d37-394b694d8ef7
md"""
!!! info "Principle"
	$$f'(x_0) \approx \frac{f(x_0 + h) - f(x_0)}{h}$$

!!! danger "Downside"
	Inexact due to approximation or floating point errors
"""

# ╔═╡ a383b9ea-3ace-470a-874c-a5f32fcaa29b
md"""
## Horseman IV: Algorithmic
"""

# ╔═╡ 42330b47-7a36-4f83-afb7-54139dacbbf2
md"""
!!! info "Principle"
	Reinterpret the source code to compute derivatives along with primal values

!!! danger "Downside"
	Hard to implement
"""

# ╔═╡ 50e1e9f5-b263-4a35-b5a1-5b6bcdf2ddad
md"""
Faster than symbolic, more precise than numerical
"""

# ╔═╡ de1424c4-ebac-4155-b6d2-489c9724245f
md"""
 $L =$ $(@bind L_slider2 PlutoUI.Slider(0:500:10^4; show_value=true))
"""

# ╔═╡ fb66a558-d4fe-41bc-a146-e71da8574f4f
md"""
# Forward and reverse mode
"""

# ╔═╡ fb641d47-2377-4d48-8723-8026f9ab8702
md"""
## Jacobians
"""

# ╔═╡ 201f3e38-37d2-4d7b-9e22-62f2ae7916a0
md"""
### Chain rule
"""

# ╔═╡ 2365e621-11a1-4945-b7c6-b82e09cf2400
md"""
Consider a vector-to-vector function
```math
f : x \in \mathbb{R}^{n_\text{in}} \longmapsto y \in \mathbb{R}^{n_\text{out}}
```
defined as the composition of $L$ elementary functions (or layers)
```math
f = f_L \circ f_{L-1} \circ \dots \circ f_2 \circ f_1
```
Its Jacobian matrix is a product, given by the chain rule
```math
J = J_L J_{L-1} \cdots J_2 J_1 \quad \text{with} \quad J_\ell = \partial f_\ell [(f_{\ell-1} \circ \dots \circ f_1)(x)] 
```
Do we need to compute full Jacobians to compose functions?
"""

# ╔═╡ 10f7588c-7bc3-4b83-a191-def09a937edd
md"""
### From matrices to linear maps
"""

# ╔═╡ eee51fe7-10c0-45ed-b2bf-35a4e10bb624
md"""
Consider Jacobians as linear maps instead:

- generalizes to arbitrary inputs / outputs (e.g. manifolds and their tangent spaces)
- more efficient in high dimension
- sufficient to solve linear systems $Ju = b$ thanks to iterative methods (like GMRES)

The maps we need are:

- **Jacobian-vector product (JVP)** or pushforward: $u \in \mathbb{R}^{n_{\text{in}}} \mapsto Ju \in \mathbb{R}^{n_{\text{out}}}$
- **Vector-Jacobian product (VJP)** or pullback: $v \in \mathbb{R}^{n_{\text{out}}} \mapsto v^\top J = J^\top v \in \mathbb{R}^{n_{\text{in}}}$

"""

# ╔═╡ 5e9bf404-d276-486a-bad1-07318b7478d2
md"""
### From linear maps back to matrices
"""

# ╔═╡ 097a0c8a-1404-4c85-ae92-1b67b9c13d4e
md"""
The Jacobian of a vector-to-vector function can be filled
  - column by column from $n_\text{in}$ JVPs, one per input basis vector $e_j$
  - row by row from $n_\text{out}$ VJPs, one per output basis vector $e_i$
"""

# ╔═╡ 06dc252d-2e83-4af5-a215-258b6709f9cf
md"""
## Forward mode
"""

# ╔═╡ 7d09d62e-8e97-4891-8628-e7a64b0b4dda
md"""
### Jacobian-vector product (JVP)
"""

# ╔═╡ f89d60cd-c59d-40c2-8531-d900f92c42d8
md"""
An input perturbation $u \in \mathbb{R}^{n_\text{in}}$ can be propagated in the forward direction ($1$ to $L$, left to right):
```math
Ju = J_L(J_{L-1} \dots (J_2(J_1 u)))
```
All we need is a JVP operator $u \mapsto J_\ell u$ for each layer $f_\ell$.

In forward mode AD, the complexity of a JVP is proportional to the complexity of one call to $f$.
"""

# ╔═╡ b8f2c1bf-fb99-457a-aac1-479b9c583351
LocalResource("images/forward.png")  # source: "The Elements of Differentiable Programming"

# ╔═╡ 5e3be8e6-611e-428c-bcc6-0da0ecdd5368
md"""
### JVP implementation
"""

# ╔═╡ 372b4a11-9056-4381-9758-c0ca0d5a3ff4
md"""
Forward mode AD is very easy to implement with "operator overloading", using dual numbers (similar to complex step).
"""

# ╔═╡ ae9445ef-c2a6-4f5b-8b48-9081e0827820
struct Dual{T<:Real}
	val::T  # store the primal value
	der::T  # store the derivative
end

# ╔═╡ 9326c8b2-4f42-4219-80e5-79b7e7c192ff
Base.:*(x1::Dual, x2::Real) = Dual(
	x1.val * x2,
	x1.der * x2
)

# ╔═╡ e410f855-2710-4856-a2ab-fee3a9cd3ce4
Base.:/(x1::Dual, x2::Real) = Dual(
	x1.val / x2,
	x1.der / x2
)

# ╔═╡ a053073d-710f-43cd-8ffe-a395f69fb5e7
Base.:+(x1::Dual, x2::Dual) = Dual(
	x1.val + x2.val,
	x1.der + x2.der
)

# ╔═╡ c2fb908a-147b-450b-8ff5-9b06f4666354
Base.:*(x1::Dual, x2::Dual) = Dual(
	x1.val * x2.val,
	x1.der * x2.val + x1.val * x2.der
)

# ╔═╡ b335cff4-54b6-4a10-bc75-cbb438ef5a64
dual_der(f, x0) = f(Dual(x0, one(x0))).der

# ╔═╡ dbe28614-186f-4a27-a36c-c94cfdfbc21a
md"""
### Derivatives
"""

# ╔═╡ 1c5ab740-dfb9-4ae4-94f2-6292202af7c8
md"""
The derivative of a scalar-to-vector function is a JVP applied to the input perturbation $u = [1]$ (efficient in forward mode)
"""

# ╔═╡ c96b3724-3c14-44fe-ab4d-aff5a0a29aa4
md"""
## Reverse mode
"""

# ╔═╡ 5a2f938e-604f-48c9-87e4-3e0a6ce8e974
md"""
### Vector-Jacobian product (VJP)
"""

# ╔═╡ 3685e527-d812-40b1-aa0e-e608818efac9
md"""
An output sensitivity $v \in \mathbb{R}^{n_\text{out}}$ can be propagated in the reverse direction ($L$ to $1$, right to left):
```math
v^\top J = (((v^\top J_L)J_{L-1}) \dots J_2)J_1
```
All we need is a VJP operator $v \mapsto v^\top J_\ell$ for each layer $f_\ell$.

In reverse mode AD, the complexity of a VJP is proportional to the complexity of one call to $f$.
"""

# ╔═╡ 63eeecbc-2e65-4993-ba61-7cd97bfd31eb
LocalResource("images/reverse.png")  # source: "The Elements of Differentiable Programming"

# ╔═╡ b851c800-0435-4d0f-a6dd-f49db8405293
md"""
### VJP implementation
"""

# ╔═╡ d57af4b9-fef0-4ca5-8c9e-c0c160f718c4
md"""
Much more complicated than forward mode due to memory requirements:

1. Run a forward sweep and record everything that happened on a tape
2. Run a reverse sweep to compute the VJP
"""

# ╔═╡ b913201b-3980-4922-bbe3-8aa8e1fc633c
LocalResource("images/memory.png")  # source: "The Elements of Differentiable Programming"

# ╔═╡ 002cc839-dc2c-454f-8a22-e62ced9f6d54
md"""
### Gradients
"""

# ╔═╡ ec7ac2ae-26f5-4b72-be84-a6c9dc1be613
md"""
The gradient of a vector-to-scalar function is a VJP applied to the output sensitivity $v = [1]$ (efficient in reverse mode)

!!! tip "Baur-Strassen theorem"
	Gradient computation is fast, so machine learning is possible
"""

# ╔═╡ 05a2f30f-a9c9-49dd-80e3-9d18a1835e97
md"""
# Clever combinations
"""

# ╔═╡ 0a53e623-8c15-44f0-947a-76438e7ef5b7
md"""
## Second order
"""

# ╔═╡ 219c9de4-0c7a-4cbe-8b74-25e7668487fa
md"""
> [How to compute Hessian-vector products?](https://iclr-blogposts.github.io/2024/blog/bench-hvp/), Dagréou et al. (2024)
"""

# ╔═╡ e0140759-c2ef-49b3-b93d-f8ef6b8aca65
md"""
A Hessian-vector product (HVP) can be computed by combining forward and reverse mode:
```math
\nabla^2 f(x)[u] = (\partial(\nabla f))(x)[u] 
```
In forward-over-reverse mode AD, the complexity of an HVP is proportional to the complexity of one call to $f$.

Sufficient to compute the Hessian entirely, or to solve linear systems $Hu = b$ with iterative methods (like conjugate gradient)
"""

# ╔═╡ e7f0fa32-449d-44e5-b985-b1e5a57f7cf7
md"""
## Sparsity
"""

# ╔═╡ a5f8cb54-e62e-4220-8da9-f16d01ebfc6d
md"""
> [What Color Is Your Jacobian? Graph Coloring for Computing Derivatives](https://epubs.siam.org/doi/abs/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""

# ╔═╡ 6db8d4c1-1699-495e-bee3-f7b66012a7ad
md"""
Real-life Jacobians and Hessians are often sparse.

Can this information speed up computation?
"""

# ╔═╡ 9700380c-ede9-4d26-ba01-b15bcf7ba070
md"""
### Compressed column evaluation
"""

# ╔═╡ 0a25735f-0f0f-4829-b944-0459ec3706ee
md"""
Jacobian in forward mode has complexity $O(n_\text{in})$: each column $J_j$ obtained as $\texttt{jvp}(e_j)$.

In the sparse case, it makes sense to evaluate several columns at once:
```math
J_{j_1} + ... + J_{j_k} = \texttt{jvp}(e_{j_1} + \dots + e_{j_k})
```
If the nonzero patterns of the columns are orthogonal, decompression is unambiguous.
"""

# ╔═╡ 1d62bfe7-c43c-4e3c-b4a8-59cf2016209f
LocalResource("images/compressed_jacobian.png")  # source: "What color is your Jacobian?"

# ╔═╡ e8f7f7b5-f24c-452a-af68-48efcb1e9e97
md"""
### Bipartite graph coloring
"""

# ╔═╡ 8e4a6323-aba4-46f5-80ef-4db54c58d912
md"""
How to find subsets of mutually orthogonal columns from the sparsity pattern $S$?

1. Build a bipartite graph $G = (R \cup C, E)$ representing $S$:
    - the vertices are the rows $R$ and the columns $C$
    - the edges are $(r, c) \in E \iff S_{rc} = 1$
2. Compute a minimum distance-2 coloring of $C$
    - no two columns that are connected to the same row can share the same color

The number of JVPs needed is the number of colors.
"""

# ╔═╡ 2d904477-65e1-4c2d-9f15-3d7dbc46bd6d
LocalResource("images/partition.png")  # source: "What color is your Jacobian?"

# ╔═╡ 06889bec-a7e6-4075-8636-dc89c249ecaf
md"""
### Sparsity pattern detection
"""

# ╔═╡ f3c6296f-5b7c-43d6-9f91-af0b193477b9
md"""
In some cases, the sparsity pattern is known ahead of time. Otherwise, can we detect it?

Yes, using operator overloading once again (see [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl)).

50x [speedup](https://github.com/adrhill/SparseConnectivityTracer.jl/pull/4) over previous Julia solution, does not even exist in [JAX](https://github.com/google/jax/issues/1032) or PyTorch
"""

# ╔═╡ ab231166-e6b5-42a7-9165-534adb03af5d
struct Tracer
	indices::Set{Int}
end

# ╔═╡ 320796bb-14f3-441a-afd0-59f9f78f7776
Base.:*(x1::Tracer, x2::Tracer) = Tracer(x1.indices ∪ x2.indices)

# ╔═╡ 29890b70-84a1-4f2f-b572-ab21d231d236
Base.:/(x1::Dual, x2::Dual) = Dual(
	x1.val / x2.val,
	(x1.der * x2.val - x2.der * x1.val) / x2.val^2
)

# ╔═╡ 7a3110f3-ca56-4be6-a9a3-7547064b3a69
function myalgo(x, L::Integer=10)
	y = x
	for l in 1:L
		y = (y + x/y) / 2
	end
	return y
end

# ╔═╡ 8f14ceb9-37fa-4afb-bbad-8d2c25e978db
myalgo(x0, 10)

# ╔═╡ 84da59f4-54a4-41ea-b0c6-26d40536a3bf
let
	x = Symbolics.variable(:x)
	@time myalgo(x, L_slider)
end

# ╔═╡ 15b4c88c-2707-4ede-b12f-174be0d179a0
let
	x = Symbolics.variable(:x)
	@time Symbolics.derivative(myalgo(x, L_slider), x)
end

# ╔═╡ 683a683c-939b-4060-a14d-a592e1e7266d
@time ForwardDiff.derivative(Base.Fix2(myalgo, L_slider2), x0)

# ╔═╡ 6a5d0cc7-483f-460e-bc07-69c19430a3ac
dual_der(myalgo, x0)

# ╔═╡ a06be073-9d10-4480-8d26-26cda704fa46
finitediff_der(f, x0; h) = (f(x0 + h) - f(x0)) / h

# ╔═╡ 7e069828-7c66-4415-aa46-d4bc49629cfc
finitediff_central_der(f, x0; h) = (f(x0 + h) - f(x0 - h)) / 2h

# ╔═╡ 70201672-dacf-4787-a33c-6756a3b6ee57
finitediff_complex_der(f, x0; h) = imag(f(x0 + h * im) - f(x0)) / h

# ╔═╡ c52d621d-9e5f-4e29-8d8c-b0a8de84d2fb
let
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="step size", ylabel="error", xscale=log10, yscale=log10)
	hs = 10.0 .^ (-15:0.2:0)
	finitediff_errs = map(
		h -> abs(finitediff_der(myalgo, x0; h) - inv(2 * sqrt(x0))), hs
	)
	finitediff_central_errs = map(
		h -> abs(finitediff_central_der(myalgo, x0; h) - inv(2 * sqrt(x0))), hs
	)
	finitediff_complex_errs = map(
		h -> abs(finitediff_complex_der(myalgo, x0; h) - inv(2 * sqrt(x0))), hs
	)
	forwarddiff_err = abs(ForwardDiff.derivative(myalgo, x0) - inv(2 * sqrt(x0)))
	l1 = scatterlines!(ax, hs, max.(eps(), finitediff_errs), color=:red)
	l2 = scatterlines!(ax, hs, max.(eps(), finitediff_central_errs), color=:blue)
	l3 = scatterlines!(ax, hs, max.(eps(), finitediff_complex_errs), color=:green)
	l4 = hlines!(ax, [max(eps(), forwarddiff_err)], color=:black, linewidth=3)
	Legend(
		fig[1,2], [l1, l2, l3, l4],
		["forward finite diff\n(real step)", "central finite diff\n(real step)", "forward finite diff\n(complex step)", "algorithmic"]
	)
	fig
end

# ╔═╡ 8160588c-5ace-4c43-b2d4-043e8173944e
inv(2 * sqrt(x0))

# ╔═╡ 5a4c9192-5c42-46fb-9384-f71b1dd4760a
prod_first_last(x) = x[begin] * x[end]

# ╔═╡ edcbf517-5d92-4152-9fc2-c6f3f864ad55
let
	x = [Tracer(Set(1)), Tracer(Set(2)), Tracer(Set(3))]
	y = prod_first_last(x)
end

# ╔═╡ 289c8a07-c29b-4ac9-8bbc-56f806fdf988
md"""
# Limits and workarounds
"""

# ╔═╡ 5106f29b-2b58-41b7-9e90-9dbd09cfcff8
md"""
Exact and approximate algorithms for differentiating through funky stuff.
"""

# ╔═╡ b665de9e-07a7-4d6c-bac6-707d3d90b9ef
md"""
## Optimization problems
"""

# ╔═╡ 27f93503-b82f-45a0-834d-7004b21282b7
md"""
> [OptNet: Differentiable Optimization as a Layer in Neural Networks](https://proceedings.mlr.press/v70/amos17a.html), Amos and Kolter (2017)
>
> [Efficient and modular implicit differentiation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html), Blondel et al. (2022)
"""

# ╔═╡ 4612b4a2-727c-44d9-b576-b284d7c30eec
md"""
Value and solution of a strongly convex optimization problem are smooth functions of its parameters.
"""

# ╔═╡ aeaaa6aa-cd58-410d-8ef2-4b36c3ae28ee
md"""
### Differentiate through a max
"""

# ╔═╡ a6a9fea0-adfa-41e0-8f27-37e53f51ea65
md"""
```math
m^\star(x) = \max_y g(x, y)
```

Exchange $\nabla$ and $\max$ using Danskin's / Rockafellar's theorem

```math
\nabla m^\star(x) = \nabla_1 g(x, y^\star(x)) \quad \text{with} \quad y^\star(x) = \arg \max_y g(x, y)
```
"""

# ╔═╡ 8b1ae57b-acc9-4d32-9e19-93b2049a9cf7
md"""
### Differentiate through an argmax
"""

# ╔═╡ d8d4cc2a-1fcd-4c59-82f7-d72c2db02596
md"""
```math
	y^\star(x) = \arg \max_y g(x, y)
```
Use the implicit function theorem with the optimality conditions $c(x, y^\star(x)) = 0$, then solve a linear system:
```math
	\underbrace{\partial_1 c(x, y^\star(x))}_{-B} + \underbrace{\partial_2 c(x, y^\star(x))}_A \cdot \underbrace{\partial y^\star(x)}_J = 0
```
What kind of optimality conditions?

- Unconstrained: gradient is zero $c(x, y) = \nabla_2 g(x, y) = 0$
- Constrained: KKT, projected gradient
"""

# ╔═╡ f2ff2731-77ca-456e-b5ca-74f2d4ea74aa
md"""
## Expectations
"""

# ╔═╡ 2684079a-ea96-4c55-b565-390a0b6d2de0
md"""
> [Monte Carlo Gradient Estimation in Machine Learning ](https://www.jmlr.org/papers/v21/19-346.html), Mohamed et al. (2020)
"""

# ╔═╡ d4799108-9061-45af-b238-6a0a945082f8
md"""
Expectations of parametric distributions
```math
f(x) = \mathbb{E}_{Y \sim p_x}[g(Y)] = \int g(y) \, p_x(y) \, \mathrm{d}y
```
are usually approximated by Monte-Carlo
```math
\widehat{f}_K(x) = \frac{1}{K} \sum_{k=1}^K g(y_k) \quad \text{with} \quad y_k \overset{\text{iid}}{\sim} p_x
```
"""

# ╔═╡ 4821a3a6-8178-4d1d-bf2e-d9f45ced4715
md"""
### REINFORCE / score function
"""

# ╔═╡ e2e55824-657d-43fb-839a-3181656620e5
md"""
Exchange $\nabla$ and $\int$, then use the logarithmic identity $\nabla_x \log p_x(y) = \frac{\nabla_x p_x(y)}{p_x(y)}$ to recover an expectation:
```math
\begin{align*}
\nabla f(x) & = \int g(y) \, \nabla_x p_x(y) \, \mathrm{d}y\\
& = \int g(y) \, p_x(y) \, \nabla_x \log p_x(y) \, \mathrm{d}y \\
& = \mathbb{E}_{Y \sim p_x} [g(Y) \nabla_x \log p_x(Y)]
\end{align*}
```
Monte-Carlo gradient estimate:
```math
\widehat{\nabla f}_K(x) = \frac{1}{K} \sum_{k=1}^K g(y_k) \nabla_x \log p_x(y_k)
```
"""

# ╔═╡ 86444d3c-3712-4c23-bfcd-b9553bc714f1
TwoColumn(
	md"""
Upsides:
- works with any $g$ and $p_x$
	""",
	md"""
Downsides:
- high variance
	"""
)

# ╔═╡ 616fe6e8-e4b8-48fb-ae3c-65c2b28106de
md"""
### Reparametrization / pathwise
"""

# ╔═╡ 24972121-e793-402e-9719-c3784711a4c3
md"""
If possible, express $Y \sim p_x$ as $Y = h(x, Z)$ with $Z \sim q$ indep from $x$:
```math
\begin{align*}
f(x) & = \int g(h(x, z)) \, q(z) \, \mathrm{d}z \\
\nabla f(x) &= \int (\partial_1 h(x, z))^\top \nabla g(h(x, z)) \, q(z) \, \mathrm{d}z
\\
& = \mathbb{E}_{Z \sim q} \left[ (\partial_1 h(x, Z))^\top \nabla g(h(x, Z)) \right]
\end{align*} 
```
Monte-Carlo gradient estimate:
```math
\widehat{\nabla f}_K(x) = \frac{1}{K} \sum_{k=1}^K (\partial_1 h(x, z_k))^\top \nabla g(h(x, z_k)) \quad \text{with} \quad z_k \overset{\text{iid}}{\sim} q
```
"""

# ╔═╡ 749cec74-b9a9-41ca-a048-e19cb241b492
TwoColumn(
	md"""
Upsides:
- low variance
	""",
	md"""
Downsides:
- reparametrization not always possible
- requires $g$ differentiable
	"""
)

# ╔═╡ e964923b-870b-4b88-a22d-d81af6caf385
md"""
### Stochastic programs
"""

# ╔═╡ d3809b7f-c832-4109-a2f6-7878d73115ed
md"""
> [Gradient Estimation Using Stochastic Computation Graphs](https://proceedings.neurips.cc/paper_files/paper/2015/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html), Schulman et al. (2015)
"""

# ╔═╡ ef2d3e57-ac3b-4b5e-a063-3eff1fdf11be
md"""
What happens when the function can return a random result?

Different from expectations which are deterministic

Requires generalizing computation graphs with "stochastic nodes"
"""

# ╔═╡ b01724db-42fd-4ebb-9d67-74d7a19bf36c
md"""
## Discrete algorithms
"""

# ╔═╡ 4b4d5e05-22ec-4093-9dfd-b5cc7191540a
md"""
### Control flow
"""

# ╔═╡ 4e7873ed-0ddd-4b7b-8acc-2d55be3257bc
md"""
Algorithmic differentiation takes the derivative for the specific control flow of the forward sweep:

- same `if/else` branching
- same number of iterations in `while` loop
- etc.

We can also define continuous relaxations of the control flow, e.g. by taking each branch with a nonzero probability.
"""

# ╔═╡ 0b520f07-8b7a-443f-9044-6dd727701c42
md"""
### Combinatorial solvers
"""

# ╔═╡ d49a9db0-2154-4a3e-8359-195830286324
md"""
> [Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities](https://arxiv.org/abs/2307.13565), Mandi et al. (2023)
"""

# ╔═╡ c6c58d3c-5346-485b-9d3e-0b1a46d8733b
md"""
An integer linear program is a piecewise-constant mapping from objective to solution:
```math
y^\star(x) = \arg \max_{y \in \mathbb{Z}} x^\top y \quad \text{s.t.} \quad Ay \leq b
```
Its derivatives contain no information, but we may want to use it within a deep learning pipeline.
"""

# ╔═╡ 88d8c6fd-fe08-45f4-ad3a-55af8e9131f5
md"""
### Regularization
"""

# ╔═╡ 5113cb7c-ff61-4b06-9561-d464aadd26f2
md"""
Apply implicit differentiation to a convex regularized version of the problem:
```math
\widehat{y^\star_R}(x) = \arg \max_{y \in \mathbb{R}} (x^\top y - R(y)) \quad \text{s.t.} \quad Ay \leq b
```
"""

# ╔═╡ b9853f7f-fdbd-401c-b95d-7947042d5f14
md"""
### Perturbation
"""

# ╔═╡ 7cb40b09-7dfc-4ab5-988e-f7d903c6d0d4
md"""
Apply REINFORCE to a randomly perturbed version of the problem:
```math
\widehat{y^\star_\varepsilon}(x) = \mathbb{E}\left[\arg \max_{y \in \mathbb{Z}} (x + \varepsilon Z)^\top y \quad \text{s.t.} \quad Ay \leq b\right] \quad \text{where} \quad Z \sim \mathcal{N}(0, \varepsilon^2)
```
"""

# ╔═╡ f6f4191f-faf6-407c-9c65-b5fd24547296
md"""
## Link with compilation
"""

# ╔═╡ 77063fdb-c980-4f09-b34f-f64b79ac6782
md"""
[Enzyme](https://github.com/EnzymeAD/enzyme): lower, optimize and then differentiate
"""

# ╔═╡ 3302c1da-30d6-4485-ac6d-16a48d10450c
md"""
The following slides are borrowed to William S. Moses

<https://indico.cern.ch/event/1145124/contributions/4994088/attachments/2508821/4311554/enzyme-mode.pdf>
"""

# ╔═╡ 04591e34-049c-484b-a018-35fc25556292
TwoColumn(
	LocalResource("images/compiler_preopt.png"),
	LocalResource("images/compiler_postopt.png"),
)

# ╔═╡ 37e08d2e-40e7-4780-b29c-4a600788f839
LocalResource("images/compiler_preopt_vs_postopt.png")

# ╔═╡ c24656e1-b5df-4eb0-b5e9-a28d50ea3bb1
LocalResource("images/enzyme_diff_opt.png")

# ╔═╡ cccd79cf-60f4-4617-8437-ab1ec5c60ca3
LocalResource("images/enzyme_opt_diff.png")

# ╔═╡ f9df653b-3b72-4d9b-9e4d-462d08043de0
md"""
# Questions?
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
CairoMakie = "~0.11.10"
ForwardDiff = "~0.10.36"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.58"
Symbolics = "~5.28.0"
Zygote = "~0.6.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "f30b366c3a6d672b079b2a9de465b515b8abc0b7"

[[deps.ADTypes]]
git-tree-sha1 = "fcdb00b4d412b80ab08e39978e3bdef579e5e224"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.0.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractLattices]]
git-tree-sha1 = "222ee9e50b98f51b5d78feb93dd928880df35f06"
uuid = "398f06c4-4d28-53ec-89ca-5b2656b7603d"
version = "0.3.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "297b6b41b66ac7cbbebb4a740844310db9fd7b8c"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown", "Test"]
git-tree-sha1 = "c0d491ef0b135fd7d63cbc6404286bc633329425"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.36"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["Random"]
git-tree-sha1 = "ca95b2220ef440817963baa71525a8f2f4ae7f8f"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.0.0"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "133a240faec6e074e07c31ee75619c90544179cf"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.10.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "588e0d680ad1d7201d4c6a804dcb1cd9cba79fbb"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.3"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bijections]]
git-tree-sha1 = "c9b163bd832e023571e86d0b90d9de92a9879088"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "e64af6bea1f0dcde6ebecc581768074f992ad39b"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.11.10"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a4c43f59baa34011e303e76f5c8c91bf58415aaf"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3e79289d94b579d81618f4c7c974bb9390dab493"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.64.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "575cd02e080939a33b6df6c5853d14924c08e35b"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.23.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelaunayTriangulation]]
deps = ["DataStructures", "EnumX", "ExactPredicates", "Random", "SimpleGraphs"]
git-tree-sha1 = "d4e9dc4c6106b8d44e40cd4faf8261a678552c7c"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "0.8.12"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "22c595ca4146c07b16bcf9c8bea86f731f7109d2"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.108"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "a0c11553b6c58bfc0283d8279dd3798150b77ac5"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.12"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "0c056035f7de73b203a5295a22137f96fc32ad46"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.5.6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ab3f7e1819dba9434a3a5126510c8fda3a4e7000"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.1+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "bfe82a708416cf00b73a3198db0859c82f741558"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.10.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "2de436b72c3422940cbe1367611d137008af7ec3"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.23.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "2493cdfd0740015955a8e46de4ef28f49460d8bc"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.3"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "68e8ff56a4a355a85d2784b94614491f8c900cde"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "10.1.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "801aef8228f7f04972e596b09d4dba481807c913"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.4"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "5694b56ccf9d15addedc35e9a4ba9c317721b788"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.10"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "359a1ba2e320790ddbe4ee8b4d54a305c0ea2aff"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.0+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "6f93a83ca11346771a93bbde2bdad2f65b61498f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.10.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "5d8c5713f38f7bc029e26627b687710ba406d0dd"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.12"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be50fe8df3acbffa0274a744f1a99d29c45a57f4"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "23ddd329f4a2a65c7a55b91553b60849bd038575"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.11"
weakdeps = ["DiffRules", "ForwardDiff", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "896385798a8d49a255c398bd49162062e4a4c435"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.13"
weakdeps = ["Dates"]

    [deps.InverseFunctions.extensions]
    DatesExt = "Dates"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "e9648d90370e2d0317f9518c9c6e0841db54a90b"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.31"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "839c82932db86740ae729779e610f07a1640be9a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.6.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "88b916503aac4fb7f701bb625cd84ca5dd1677bc"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.29+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "LinearAlgebra", "MacroTools", "PreallocationTools", "RecursiveArrayTools", "StaticArrays"]
git-tree-sha1 = "d1f981fba6eb3ec393eede4821bca3f2b7592cd4"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.15.1"

[[deps.LambertW]]
git-tree-sha1 = "c5ffc834de5d61d00d2b0e18c96267cffc21f648"
uuid = "984bce1d-4616-540c-a9ee-88d1112d94c9"
version = "0.4.6"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "e0b5cd21dc1b44ec6e64f351976f961e6f31d6c4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.3"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dae976433497a2f841baadea93d27e68f1a12a97"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.39.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0a04a1318df1bf510beb2562cf90fb0c386f58c4"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.39.3+1"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "3a994404d3f6709610701c7dabfc03fed87a81f8"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearAlgebraX]]
deps = ["LinearAlgebra", "Mods", "Primes", "SimplePolynomials"]
git-tree-sha1 = "d76cec8007ec123c2b681269d40f94b053473fcf"
uuid = "9b3f67b0-2d00-526e-9884-9e4938f8fb88"
version = "0.2.7"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "31e27f0b0bf0df3e3e951bfcc43fe8c730a219f6"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.4.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "80b2833b56d466b3858d565adcd16a4a05f2089b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.1.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "46ca613be7a1358fb93529726ea2fc28050d3ae0"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.20.9"

[[deps.MakieCore]]
deps = ["Observables", "REPL"]
git-tree-sha1 = "248b7a4be0f92b497f7a331aed02c1e9a878f46b"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.7.3"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "96ca8a313eb6437db5ffe946c457a401bbb8ce1d"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mods]]
git-tree-sha1 = "924f962b524a71eef7a21dae1e6853817f9b658f"
uuid = "7475f97c-0381-53b1-977b-4c60186c8d62"
version = "2.2.4"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.Multisets]]
git-tree-sha1 = "8d852646862c96e226367ad10c8af56099b4047e"
uuid = "3b2b4ff1-bcff-5658-a3ee-dbcf1ce5ac09"
version = "0.4.4"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "769c9175942d91ed9b83fa929eee4fe6a1d128ad"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "c2e2d748aea87f006a87c1654878349baa04aaf0"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.3"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "e64b4f5ea6b7389f6f046d13d4896a8f9c1ba71e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3da7367955dcc5c54c1ba4d402ccdc09a1a3e046"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cb5a2ab6763464ae0f19c86c56c63d4a2b0f5bda"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.52.2+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Permutations]]
deps = ["Combinatorics", "LinearAlgebra", "Random"]
git-tree-sha1 = "eb3f9df2457819bf0a9019bd93cc451697a0751e"
uuid = "2ae35dd2-176d-5d53-8349-f30d82d94d4f"
version = "0.4.20"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "89f57f710cc121a7f32473791af3d6beefc59051"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.14"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "81a2a9462003a423fdc59e2a3ff84cde93c4637b"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.7"
weakdeps = ["ChainRulesCore", "FFTW", "MakieCore", "MutableArithmetics"]

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff"]
git-tree-sha1 = "a660e9daab5db07adf3dedfe09b435cc530d855e"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.21"

    [deps.PreallocationTools.extensions]
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "cb420f77dc474d23ee47ca8d14c90810cafe69e7"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.6"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "763a8ceb07833dd51bb9e3bbca372de32c0605ad"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "SparseArrays", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "d8f131090f2e44b145084928856a561c83f43b27"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.13.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "12aa2d7593df490c407a3bbd8b86b8b515017f3e"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.14"

[[deps.RingLists]]
deps = ["Random"]
git-tree-sha1 = "f39da63aa6d2d88e0c1bd20ed6a3ff9ea7171ada"
uuid = "286e9d63-9694-5540-9e3c-4e6708fa07b2"
version = "0.2.8"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "e86ff73265bc346b964cb5209080f18b9a6d7edf"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.34.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools", "Setfield", "SparseArrays", "StaticArraysCore"]
git-tree-sha1 = "10499f619ef6e890f3f4a38914481cc868689cd5"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.8"

[[deps.SciMLStructures]]
git-tree-sha1 = "5833c10ce83d690c124beedfe5f621b50b02ba4d"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.1.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleGraphs]]
deps = ["AbstractLattices", "Combinatorics", "DataStructures", "IterTools", "LightXML", "LinearAlgebra", "LinearAlgebraX", "Optim", "Primes", "Random", "RingLists", "SimplePartitions", "SimplePolynomials", "SimpleRandom", "SparseArrays", "Statistics"]
git-tree-sha1 = "f65caa24a622f985cc341de81d3f9744435d0d0f"
uuid = "55797a34-41de-5266-9ec1-32ac4eb504d3"
version = "0.8.6"

[[deps.SimplePartitions]]
deps = ["AbstractLattices", "DataStructures", "Permutations"]
git-tree-sha1 = "e182b9e5afb194142d4668536345a365ea19363a"
uuid = "ec83eff0-a5b5-5643-ae32-5cbf6eedec9d"
version = "0.3.2"

[[deps.SimplePolynomials]]
deps = ["Mods", "Multisets", "Polynomials", "Primes"]
git-tree-sha1 = "7063828369cafa93f3187b3d0159f05582011405"
uuid = "cc47b68c-3164-5771-a705-2bc0097375a0"
version = "0.2.17"

[[deps.SimpleRandom]]
deps = ["Distributions", "LinearAlgebra", "Random"]
git-tree-sha1 = "3a6fb395e37afab81aeea85bae48a4db5cd7244a"
uuid = "a6525b86-64cd-54fa-8f65-62fc48bdc0e8"
version = "0.3.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"
weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "MacroTools", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "40ea524431a92328cd73582d1820a5b08247a40f"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.16"

[[deps.SymbolicLimits]]
deps = ["SymbolicUtils"]
git-tree-sha1 = "89aa6b25a75418c8fffc42073b2e7dce69847394"
uuid = "19f23fe9-fdab-4a78-91af-e7b7767979c3"
version = "0.2.0"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "TimerOutputs", "Unityper"]
git-tree-sha1 = "669e43e90df46fcee4aa859b587da7a7948272ac"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "1.5.1"

[[deps.Symbolics]]
deps = ["ArrayInterface", "Bijections", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "ForwardDiff", "IfElse", "LaTeXStrings", "LambertW", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "PrecompileTools", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "SymbolicLimits", "SymbolicUtils"]
git-tree-sha1 = "4104548fff14d7370b278ee767651d6ec61eb195"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "5.28.0"

    [deps.Symbolics.extensions]
    SymbolicsGroebnerExt = "Groebner"
    SymbolicsLuxCoreExt = "LuxCore"
    SymbolicsPreallocationToolsExt = "PreallocationTools"
    SymbolicsSymPyExt = "SymPy"

    [deps.Symbolics.weakdeps]
    Groebner = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
    LuxCore = "bb33d45b-7691-41d6-9220-0943567d0623"
    PreallocationTools = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "25008b734a03736c41e2a7dc314ecb95bd6bbdb0"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "532e22cf7be8462035d092ff21fada7527e2c488"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.6+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "4ddb4470e47b0094c93055a3bcae799165cc68f1"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.69"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╠═028bee0c-00d6-11ef-2159-99ffeebdeba2
# ╠═cbf0a7eb-0750-4537-b925-efd743360979
# ╟─2fe59b4a-3bd4-4466-83ff-aa97a887b1b3
# ╟─fe638244-3f35-4de4-a52e-aa6ef8b662ef
# ╟─d98a6acc-8cb9-4ec3-beb2-c12655f0bd72
# ╟─824423d7-21b0-4b48-9200-22d213207fe0
# ╟─603d7e48-3def-44f8-882e-74e3f22293c0
# ╟─fdb8ed44-74dd-4cdf-8d3d-c4d8bf467fc4
# ╟─317749dd-fd49-488b-ad0d-9d30b33d9891
# ╟─7b023270-9b2d-4d69-93dc-921e37fcf423
# ╠═ff41d45f-12d8-41de-bf19-e290a0590098
# ╟─b7227f5c-bf16-460d-b2f2-6938a2ca8261
# ╟─7b64bd44-f261-49b6-8520-ec09135305d3
# ╟─52de1ea5-e8f0-4efb-a752-fdddf29ff462
# ╟─5dda51ca-cf43-4ff5-931c-258ed1268a4e
# ╟─61d3efb7-39e8-4c4b-ad8b-fd3920e8457c
# ╟─5198f895-9bf4-4f5c-ad56-109f856f4f06
# ╟─07cb8a3f-2627-47ad-bccc-abf06b994787
# ╟─ebf1147e-4469-4b22-99d2-003c3d597890
# ╟─a2030c95-7ed4-4690-9a22-3935f7f7ab47
# ╟─1298e858-8586-4589-b853-c7165a5326ed
# ╟─2d48dff9-cf14-4ede-9340-8c2d813ef8b8
# ╠═7a3110f3-ca56-4be6-a9a3-7547064b3a69
# ╠═086435e8-2d8e-49a6-a2ec-505076f0d35b
# ╠═8f14ceb9-37fa-4afb-bbad-8d2c25e978db
# ╠═129bffac-2676-453e-a8bb-404649005625
# ╟─bc8eda8e-3d2d-4939-9e3c-e0cb122fa7e8
# ╟─a3c02f24-2eb6-4895-bfd5-e95a26ac44fe
# ╟─4efaf69a-ecb3-47ad-9d00-60cd6e49d1b4
# ╟─8b4aa562-9443-40ee-8d52-c85d5cb1ef81
# ╟─cc73da3a-27ce-4963-9e28-775b34d70c29
# ╠═84da59f4-54a4-41ea-b0c6-26d40536a3bf
# ╠═15b4c88c-2707-4ede-b12f-174be0d179a0
# ╟─f5bdc5ca-a291-4244-91e1-80f6a591e5af
# ╟─b4f82186-70ae-4f4f-9d37-394b694d8ef7
# ╠═a06be073-9d10-4480-8d26-26cda704fa46
# ╠═7e069828-7c66-4415-aa46-d4bc49629cfc
# ╠═70201672-dacf-4787-a33c-6756a3b6ee57
# ╠═c52d621d-9e5f-4e29-8d8c-b0a8de84d2fb
# ╟─a383b9ea-3ace-470a-874c-a5f32fcaa29b
# ╟─42330b47-7a36-4f83-afb7-54139dacbbf2
# ╟─50e1e9f5-b263-4a35-b5a1-5b6bcdf2ddad
# ╟─de1424c4-ebac-4155-b6d2-489c9724245f
# ╠═683a683c-939b-4060-a14d-a592e1e7266d
# ╟─fb66a558-d4fe-41bc-a146-e71da8574f4f
# ╟─fb641d47-2377-4d48-8723-8026f9ab8702
# ╟─201f3e38-37d2-4d7b-9e22-62f2ae7916a0
# ╟─2365e621-11a1-4945-b7c6-b82e09cf2400
# ╟─10f7588c-7bc3-4b83-a191-def09a937edd
# ╟─eee51fe7-10c0-45ed-b2bf-35a4e10bb624
# ╟─5e9bf404-d276-486a-bad1-07318b7478d2
# ╟─097a0c8a-1404-4c85-ae92-1b67b9c13d4e
# ╟─06dc252d-2e83-4af5-a215-258b6709f9cf
# ╟─7d09d62e-8e97-4891-8628-e7a64b0b4dda
# ╟─f89d60cd-c59d-40c2-8531-d900f92c42d8
# ╠═b8f2c1bf-fb99-457a-aac1-479b9c583351
# ╟─5e3be8e6-611e-428c-bcc6-0da0ecdd5368
# ╟─372b4a11-9056-4381-9758-c0ca0d5a3ff4
# ╠═ae9445ef-c2a6-4f5b-8b48-9081e0827820
# ╠═9326c8b2-4f42-4219-80e5-79b7e7c192ff
# ╠═e410f855-2710-4856-a2ab-fee3a9cd3ce4
# ╠═a053073d-710f-43cd-8ffe-a395f69fb5e7
# ╠═c2fb908a-147b-450b-8ff5-9b06f4666354
# ╠═29890b70-84a1-4f2f-b572-ab21d231d236
# ╠═b335cff4-54b6-4a10-bc75-cbb438ef5a64
# ╠═6a5d0cc7-483f-460e-bc07-69c19430a3ac
# ╠═8160588c-5ace-4c43-b2d4-043e8173944e
# ╟─dbe28614-186f-4a27-a36c-c94cfdfbc21a
# ╟─1c5ab740-dfb9-4ae4-94f2-6292202af7c8
# ╟─c96b3724-3c14-44fe-ab4d-aff5a0a29aa4
# ╟─5a2f938e-604f-48c9-87e4-3e0a6ce8e974
# ╟─3685e527-d812-40b1-aa0e-e608818efac9
# ╠═63eeecbc-2e65-4993-ba61-7cd97bfd31eb
# ╟─b851c800-0435-4d0f-a6dd-f49db8405293
# ╟─d57af4b9-fef0-4ca5-8c9e-c0c160f718c4
# ╠═b913201b-3980-4922-bbe3-8aa8e1fc633c
# ╟─002cc839-dc2c-454f-8a22-e62ced9f6d54
# ╟─ec7ac2ae-26f5-4b72-be84-a6c9dc1be613
# ╟─05a2f30f-a9c9-49dd-80e3-9d18a1835e97
# ╟─0a53e623-8c15-44f0-947a-76438e7ef5b7
# ╟─219c9de4-0c7a-4cbe-8b74-25e7668487fa
# ╟─e0140759-c2ef-49b3-b93d-f8ef6b8aca65
# ╟─e7f0fa32-449d-44e5-b985-b1e5a57f7cf7
# ╟─a5f8cb54-e62e-4220-8da9-f16d01ebfc6d
# ╟─6db8d4c1-1699-495e-bee3-f7b66012a7ad
# ╟─9700380c-ede9-4d26-ba01-b15bcf7ba070
# ╟─0a25735f-0f0f-4829-b944-0459ec3706ee
# ╠═1d62bfe7-c43c-4e3c-b4a8-59cf2016209f
# ╟─e8f7f7b5-f24c-452a-af68-48efcb1e9e97
# ╟─8e4a6323-aba4-46f5-80ef-4db54c58d912
# ╠═2d904477-65e1-4c2d-9f15-3d7dbc46bd6d
# ╟─06889bec-a7e6-4075-8636-dc89c249ecaf
# ╟─f3c6296f-5b7c-43d6-9f91-af0b193477b9
# ╠═ab231166-e6b5-42a7-9165-534adb03af5d
# ╠═320796bb-14f3-441a-afd0-59f9f78f7776
# ╠═5a4c9192-5c42-46fb-9384-f71b1dd4760a
# ╠═edcbf517-5d92-4152-9fc2-c6f3f864ad55
# ╟─289c8a07-c29b-4ac9-8bbc-56f806fdf988
# ╟─5106f29b-2b58-41b7-9e90-9dbd09cfcff8
# ╟─b665de9e-07a7-4d6c-bac6-707d3d90b9ef
# ╟─27f93503-b82f-45a0-834d-7004b21282b7
# ╟─4612b4a2-727c-44d9-b576-b284d7c30eec
# ╟─aeaaa6aa-cd58-410d-8ef2-4b36c3ae28ee
# ╟─a6a9fea0-adfa-41e0-8f27-37e53f51ea65
# ╟─8b1ae57b-acc9-4d32-9e19-93b2049a9cf7
# ╟─d8d4cc2a-1fcd-4c59-82f7-d72c2db02596
# ╟─f2ff2731-77ca-456e-b5ca-74f2d4ea74aa
# ╟─2684079a-ea96-4c55-b565-390a0b6d2de0
# ╟─d4799108-9061-45af-b238-6a0a945082f8
# ╟─4821a3a6-8178-4d1d-bf2e-d9f45ced4715
# ╟─e2e55824-657d-43fb-839a-3181656620e5
# ╟─86444d3c-3712-4c23-bfcd-b9553bc714f1
# ╟─616fe6e8-e4b8-48fb-ae3c-65c2b28106de
# ╟─24972121-e793-402e-9719-c3784711a4c3
# ╟─749cec74-b9a9-41ca-a048-e19cb241b492
# ╟─e964923b-870b-4b88-a22d-d81af6caf385
# ╟─d3809b7f-c832-4109-a2f6-7878d73115ed
# ╟─ef2d3e57-ac3b-4b5e-a063-3eff1fdf11be
# ╟─b01724db-42fd-4ebb-9d67-74d7a19bf36c
# ╟─4b4d5e05-22ec-4093-9dfd-b5cc7191540a
# ╟─4e7873ed-0ddd-4b7b-8acc-2d55be3257bc
# ╟─0b520f07-8b7a-443f-9044-6dd727701c42
# ╟─d49a9db0-2154-4a3e-8359-195830286324
# ╟─c6c58d3c-5346-485b-9d3e-0b1a46d8733b
# ╟─88d8c6fd-fe08-45f4-ad3a-55af8e9131f5
# ╟─5113cb7c-ff61-4b06-9561-d464aadd26f2
# ╟─b9853f7f-fdbd-401c-b95d-7947042d5f14
# ╟─7cb40b09-7dfc-4ab5-988e-f7d903c6d0d4
# ╟─f6f4191f-faf6-407c-9c65-b5fd24547296
# ╟─77063fdb-c980-4f09-b34f-f64b79ac6782
# ╟─3302c1da-30d6-4485-ac6d-16a48d10450c
# ╟─04591e34-049c-484b-a018-35fc25556292
# ╟─37e08d2e-40e7-4780-b29c-4a600788f839
# ╟─c24656e1-b5df-4eb0-b5e9-a28d50ea3bb1
# ╟─cccd79cf-60f4-4617-8437-ab1ec5c60ca3
# ╟─f9df653b-3b72-4d9b-9e4d-462d08043de0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
