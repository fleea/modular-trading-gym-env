# Observations

Table of Contents

- [BaseObservation](#baseobservation)
- [TrendObservation](#trendobservation)
- [TrendObservation Percentage](#trendobservation-percentage)
- [TrendObservation Percentage Array Stock](#trendobservation-percentage-array-stock)
- [TrendObservation RMS](#trendobservation-rms)
- [PriceObservation](#priceobservation)
- [HLCObservation](#hlcobservation)

## BaseObservation

BaseObservation is a Protocol class that defines the minimal interface for an observation. It requires two methods to be implemented: `get_space()` and `get_observation()`. The third method, `get_start_index()`, has a default implementation that returns 0.

## TrendObservation

TrendObservation is an implementation of BaseObservation that calculates the trend of the price. It requires hyperparameters `trend_offsets` to be defined, which is an array of integers representing the offset of the trend calculation.

For example, if `trend_offsets` is [1, 2, 3], the observation will be the trend of the price at time `t-1`, `t-2`, and `t-3`.

The trend is calculated as the difference between the current price and the price at the offset. We define the start index as the maximum of the trend offsets, the training will be started from the start index.

#### Observation Vector Definition

At each discrete time step $t$, the agent receives an observation vector $\mathbf{O}\_t \in \mathbb{R}^{n+1}$, where $n$ is the number of specified trend offsets. The observation vector comprises two components:

1. **Order Indicator $o_t$**: A binary variable representing the presence of open orders at time $t$.
2. **Trend Values $\mathbf{v}\_t$**: A vector of price differences over specified time offsets.

Mathematically, the observation vector is defined as:

$$
\mathbf{O}_t = \begin{bmatrix}
o_t \\
\mathbf{v}_t
\end{bmatrix}
= \begin{bmatrix}
o_t \\
v_t^{(\tau_1)} \\
v_t^{(\tau_2)} \\
\vdots \\
v_t^{(\tau_n)}
\end{bmatrix}
\in \mathbb{R}^{n+1}
$$

#### Order Indicator $o_t$

The order indicator $o_t$ is defined as:

$$
o_t = \begin{cases}
1, & \text{if there are open orders at time } t \\
0, & \text{otherwise}
\end{cases}
$$

This component informs the agent about its current engagement in the market.

TODO: Make this a normalized float between 0 and 1 that indicate the current position where 0 is no position and 1 is maximum position that can be taken based on max_order parameter.

#### 2.2. Trend Values $\mathbf{v}\_t$

For a set of trend offsets $\{\tau*1, \tau_2, \dots, \tau_n\}$, each trend value $v_t^{(\tau_i)}$ is calculated as the difference between the current price $P_t$ and the historical price $P*{t - \tau_i}$:

$$
v_t^{(\tau_i)} = \begin{cases}
P_t - P_{t - \tau_i}, & \text{if } t - \tau_i \geq t_0 \\
0, & \text{otherwise}
\end{cases}
$$

where:

- $P_t$ is the current price at time $t$.
- $P\_{t - \tau_i}$ is the historical price at time $t - \tau_i$.
- $t*0$ is the starting index of valid data to ensure that $P*{t - \tau_i}$ exists.

This formulation captures the price movement over different time horizons, providing the agent with information on market trends.

#### Lower and Upper Bounds

- **Order Indicator Bounds**:

  $$
  l_0 = 0, \quad u_0 = 1
  $$

  $o_t$ is a binary variable, so its bounds are 0 and 1.

- **Trend Value Bounds**:
  $$
  l_i = -\infty, \quad u_i = \infty \quad \text{for } i = 1, 2, \dots, n
  $$
  Trend values can range over all real numbers due to price fluctuations.

#### Observation Space Specification

$$
\mathcal{O} = [0, 1] \times \mathbb{R}^n
$$

This defines a space where the first component is a binary variable, and the subsequent components are unbounded real numbers.

#### Starting Index $t_0$

To ensure that historical prices $P\_{t - \tau_i}$ are available for trend calculations, the starting index $t_0$ is set to:

$$
t_0 = \max\{\tau_i \mid i = 1, 2, \dots, n\}
$$

This guarantees that for all $t \geq t*0$, $P*{t - \tau_i}$ exists for each $\tau_i$.

#### Appendix: Notation

- $\mathbf{O}\_t$: Observation vector at time $t$.
- $o_t$: Order indicator at time $t$.
- $\mathbf{v}\_t$: Vector of trend values at time $t$.
- $v_t^{(\tau_i)}$: Trend value for offset $\tau_i$ at time $t$.
- $P_t$: Current price at time $t$.
- $P\_{t - \tau_i}$: Historical price at time $t - \tau_i$.
- $\tau_i$: Trend offset, a positive integer.
- $n$: Number of trend offsets.
- $\mathcal{O}$: Observation space.
- $l_i, u_i$: Lower and upper bounds for the $i$-th component.
- $t_0$: Starting index for valid data.

## TrendObservation Percentage

This observation extends the previous `TrendObservation` by calculating the percentage difference between the current price and past prices at specified offsets, instead of absolute differences.

## TrendObservation Percentage Array Stock

Analogous to `TrendObservationPercentage`, but for stock data (close value instead of bid price)

## TrendObservation RMS

In the `TrendObservationRMS` implementation, we introduce a scaling multiplier $m_t$ to normalize calculated trend. The rest of the observation vector and calculations remain analogus to `TrendObservation`.

#### Trend History and RMS Calculation

Let $h$ be the history size (e.g., $h = 30$) and $n$ be the number of trend offsets. For each time step $t$, we maintain a history of past trend values:

$$
H*t = \left\{ v*{t - k}^{(\tau_i)} \ \big| \ k = 1, 2, \dots, h; \ i = 1, 2, \dots, n \right\}
$$

Flattening the trend values over time and offsets, we compute the RMS of the trend history:

$$
\text{RMS}_t = \sqrt{ \dfrac{1}{h n} \sum_{k=1}^{h} \sum*{i=1}^{n} \left( v*{t - k}^{(\tau_i)} \right)^2 }
$$

This RMS value represents the average magnitude of the recent trend changes.

#### Multiplier Calculation

A scaling multiplier $m_t$ is calculated to normalize the trend values. The multiplier is inversely proportional to the RMS and scaled by a constant factor $\gamma$ (rms_multiplier):

$$
m_t = \begin{cases}
\displaystyle \frac{\gamma}{\text{RMS}\_t}, & \text{if } \text{RMS}\_t \neq 0 \\
1, & \text{if } \text{RMS}\_t = 0
\end{cases}
$$

#### Scaled Trend Values

The original trend values $v_t^{(\tau_i)}$ are then scaled using the multiplier to obtain the normalized trend values $\tilde{v}\_t^{(\tau_i)}$:

$$
\tilde{v}\_t^{(\tau_i)} = v_t^{(\tau_i)} \times m_t
$$

## PriceObservation

## HLCObservation

Coupled with `src/preprocessings/hlc.py`
Precompute difference between each of the high, low, close for previous day, week and month. Will be expanded with trend offset for day, week and month.
