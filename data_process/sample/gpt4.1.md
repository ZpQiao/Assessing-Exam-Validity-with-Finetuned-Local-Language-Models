未选择任何文件未选择任何文件
1,"A specific geographical area has a population of 1 million, 5% belong to a certain ethnic group. A group of 100 people is selected at random for a survey.",1,"The probability that there are at least 7 persons from the ethnic group in the sample is","1 - Φ(√(2/19))","1 - Φ(√(3/18))","1/12","1 - sum_{i=0}^6 binom{100}{i}(0.05)^i(0.95)^{100-i}","1 - sum_{i=0}^6 (e^{-5}5^i/i!)","Do not know"

Let's break down the problem:

Population = 1,000,000
Proportion of ethnic group = 5% = 0.05
Sample size = 100
Let ( X ) be the number in the sample from the ethnic group. ( X \sim \operatorname{Bin}(100, 0.05) ).
Question: Probability that at least 7 in the sample belong to the ethnic group, i.e., ( P(X \geq 7) ).
This is a binomial probability.

Step 1: Exact Expression
The exact probability (before using approximations) is:
[
P(X \geq 7) = 1 - P(X \leq 6)
]
[
= 1 - \sum_{i=0}^6 \binom{100}{i} (0.05)^i (0.95)^{100-i}
]

This matches option (d):

"1 - ( \sum_{i=0}^6 \binom{100}{i}(0.05)^i(0.95)^{100-i} )"

Step 2: Poisson Approximation
Since ( n ) is large, ( p ) is small, the Poisson approximation can be used:

Poisson mean ( \lambda = np = 5 ) [ P(X \geq 7) = 1 - P(X \leq 6) = 1 - \sum_{i=0}^{6} e^{-5} \frac{5^i}{i!} ] This matches option (e).
Step 3: Normal Approximation
For ( X \sim \mathrm{Bin}(100, 0.05) ), mean ( = 5 ), variance ( = 4.75 ).
Continuity correction: ( P(X \geq 7) \approx P\left(Z \geq \frac{6.5 - 5}{\sqrt{4.75}}\right) )

( \frac{6.5-5}{\sqrt{4.75}} \approx 0.687 )
So, ( P \approx 1 - \Phi(0.687) ).
Option (a): 1 - Φ(√(2/19))

√(2/19) ≈ 0.324 -> Doesn't match the calculated value.
Option (b): 1 - Φ(√(3/18))

√(3/18) ≈ 0.408 -> Also does not match.
Option Review
Option (d): Correct for exact Binomial solution
Option (e): Correct Poisson approximation
Therefore, the closest correct answer for the probability is:

Answer:
1 - ( \sum_{i=0}^6 \binom{100}{i} (0.05)^i (0.95)^{100-i} )

(Option d) is the exact answer.

(Option e is a Poisson approximation, which is reasonably accurate.)