<script type="text/javascript" async src="http://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$'], ['\\(','\\)']]}});</script>

<!-- MarkdownTOC autolink="true" bracket="round" markdown_preview="markdown" -->

# Markov

## 1. Markov Process(MP)
  
MDP(Markov Decision Process)에 비해 좀 더 간단한(기본이 되는) 모델.

MP problem을 Markov Chain(MC)이라고 부르기도 함.

MC는 이산 확률 프로세스(Discrete Stochastic Process), Continous Stochastic Process를 다루는 MC가 있긴하다.
-   Stochastic Process: 확률 분포를 가진 랜덤변수(random variable)가 일정한 시간 간격(Time interval)으로 값을 발생시켜 모델링하는 것.
-   이러한 모델 중 현재의 상태가 오로지 *이전 상태에만 영향을 받는 확률 프로세스* $\Rightarrow$ **Markov Process**

-   MP 모델은 두 가지 속성으로 표현 가능
    -   $X$: (유한한) 상태 공간(state space)의 집합.
    -   $P$: 전이 확률(Transition probability) = 모든 상태 $X$ 사이의 전이 확률을 의미.
<p align="center"> 
    MP(X, P)
</p>

-   Step
    -   각 state의 transition는 이산시간(Discrete time)에 이루어지며, 상태 집합 $X$에 속하는 어떤 임의의 상태에 머무를 때는 시간.
    -   현재 step이 $n$이라 하면, 다음 step은 $n+1$ 라고 기술.
-   State Transition Probability
    -   $p_{ij}$는 상태 $i$에서 




# Reference
[1] https://norman3.github.io/rl/docs/chapter01
