# Individual Project: Quantum Algorithm for Linear System Solving (HHL)

**Deadline:** 14-01-26  
**Weight:** 40% of final mark

## Overview

This project focuses on the Harrow-Hassidim-Lloyd (HHL) algorithm for quantum linear system solving, as covered in Lecture 9 [Week 10].

**Reference:** https://arxiv.org/abs/0811.3171

## Problem Setup

Consider a Hermitian matrix $A \in \mathbb{C}^{N \times N}$ with:
- Eigenvalues $\{\lambda_j\}_{j=1}^N$ satisfying $\lambda_j^{-1} < 1$ for all $j$
- Corresponding eigenvectors $\{|\psi_j\rangle\}_{j=1}^N$

Through Hamiltonian Simulation, we construct a circuit implementing:

$$U_A = e^{iAt} = \sum_{i=1}^N e^{i\lambda_i t} |\psi_i\rangle\langle\psi_i|$$

### Assumptions
- We can implement $U_A$ exactly (for simplicity)
- Since $\lambda_j^{-1} < 1$, we use $t = 1$
- Eigenvalues have exact binary representations within QPE precision

## HHL Algorithm Steps

Given $|b\rangle = \sum_{j=1}^N \beta_j |\psi_j\rangle$ expressed in the eigenbasis of $A$:

1. **Quantum Phase Estimation:** Apply QPE of $U_A$ onto $|b\rangle$ to obtain:
   $$|\phi_1\rangle = \sum_{j=1}^N \beta_j |\psi_j\rangle |\tilde{\lambda}_j\rangle$$
   where $\tilde{\lambda}_j$ is an exact binary representation of $\lambda_j$

2. **Amplitude Imprinting:** Add ancillary qubit with controlled rotation:
   $$|x\rangle|0\rangle \mapsto |x\rangle\left(\frac{1}{x}|0\rangle + \sqrt{1-|x^{-1}|^2}|1\rangle\right)$$

3. **Prepare State:**
   $$\sum_{j=1}^N \beta_j |\psi_j\rangle |\tilde{\lambda}_j\rangle \left(\frac{1}{\lambda_j}|0\rangle + \sqrt{1-|\frac{1}{\lambda_j}|^2}|1\rangle\right) \quad \text{(Eq. 1)}$$

4. **Measurement/Amplitude Amplification:** Measure ancillary register to obtain branch with ancilla in state $|0\rangle$

5. **Uncomputation:** Uncompute the QPE

**Key Insight:** The inverse of matrix $A$ is computed via the controlled rotation, where eigenvalue $\lambda$ leads to amplitude $1/\lambda$ in the $|0\rangle$ branch.

## Exercises

### Question 1 [1 point]
Expand the expression in Eq. (1) by distributivity to obtain the form:
$$|\phi_0\rangle|0\rangle + |\phi_1\rangle|1\rangle$$

Provide explicit expressions for the unnormalized vectors $|\phi_0\rangle$ and $|\phi_1\rangle$.

### Question 2 [1 point]
Using Question 1, compute the probability of obtaining outcome $|0\rangle$ when measuring the ancillary qubit (express as a function of the eigenvalues).

### Question 3 [1 point]
Consider an invertible positive semidefinite matrix $A$ with spectral decomposition $A = UDU^\dagger$, where:
$$D = \text{diag}(\lambda_1, \ldots, \lambda_N) \quad \text{with } \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_N$$

**Low-rank approximation** of rank $r$: $A' = UD'U^\dagger$ with $D' = \text{diag}(\lambda_1, \ldots, \lambda_r, 0, \ldots, 0)$

Alternatively, set threshold $\eta$ and zero all eigenvalues strictly below $\eta$.

*Application:* Low-rank approximations are crucial in signal processing for noise reduction (e.g., Netflix recommendation engine).

#### (a)
Given sparse access to $A$ and input state $|b\rangle$, how would you modify the HHL algorithm to approximately output $A'|b\rangle$, where $A'$ is the low-rank approximation with threshold $\eta$?

#### (b)
What is the success probability in the measurement step for low-rank approximations of $A$, applied to a general state $|b\rangle$?

**Hint:** Express $|b\rangle$ in the eigenbasis of $A$.
